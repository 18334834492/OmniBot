"""
OmniBot 2.0 — 全能灵控 · 主入口

跨平台AI智能体：看懂屏幕 → 规划操作 → 控制设备
2.0 架构：插件化 + 事件驱动 + 自愈重试 + 能力注册

用法:
    python -m omnibot --task "打开Chrome，搜索天气，截屏保存到桌面"
    python -m omnibot --resume checkpoints/checkpoint_step3_xxx.json
    python -m omnibot --interactive
    python -m omnibot --list-tools
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from omnibot import __version__
from omnibot.infrastructure import (
    PLATFORM, LLM_MODEL, LLM_PROVIDER, SAFE_MODE, ENABLE_CHECKPOINT,
    MEMORY_ENABLED, LOG_DIR,
)

# rich 延迟导入（可选依赖）
try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    class Console:
        def print(self, *args, **kwargs):
            print(*args, **kwargs)


console = Console() if RICH_AVAILABLE else Console()
logger = logging.getLogger("OmniBot")


# ============================================================
# 日志
# ============================================================

def _setup_logger():
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        str(LOG_DIR / f"omnibot_{datetime.now().strftime('%Y%m%d')}.log"),
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ============================================================
# Banner
# ============================================================

def print_banner():
    if RICH_AVAILABLE:
        console.print(f"""
    ╔══════════════════════════════════════════╗
    ║       OmniBot 2.0  全能灵控              ║
    ║   跨平台AI智能体 · 自然语言操控电脑        ║
    ╚══════════════════════════════════════════╝
    """, style="bold cyan")
        console.print(f"  版本: {__version__}  |  系统: {PLATFORM}  |  模型: {LLM_MODEL}  |  提供商: {LLM_PROVIDER}")
        console.print(f"  安全: {'✅' if SAFE_MODE else '❌'}  |  断点: {'✅' if ENABLE_CHECKPOINT else '❌'}  |  记忆: {'✅' if MEMORY_ENABLED else '❌'}")
        console.print()
    else:
        print(f"OmniBot {__version__} 全能灵控")
        print(f"  系统: {PLATFORM}  |  模型: {LLM_MODEL}  |  提供商: {LLM_PROVIDER}")
        print(f"  安全: {'开启' if SAFE_MODE else '关闭'}  |  断点: {'开启' if ENABLE_CHECKPOINT else '关闭'}  |  记忆: {'开启' if MEMORY_ENABLED else '关闭'}")
        print()


# ============================================================
# 运行
# ============================================================

def run_task(task: str, resume_from: str | None = None):
    """执行单个任务"""
    print_banner()

    # 初始化组件
    from omnibot.decision import (
        LLMClient, create_builtin_registry, ReActEngine,
        ToolRegistry,
    )
    from omnibot.infrastructure import (
        Guardian, CheckpointManager, get_bus, MemorySystem,
        PluginManager, EventType,
    )

    bus = get_bus()
    guard = Guardian(event_bus=bus)
    checkpoint_mgr = CheckpointManager()
    memory = MemorySystem()
    registry = create_builtin_registry()

    # 加载插件
    plugin_mgr = PluginManager(registry)
    plugin_count = plugin_mgr.auto_load()

    llm = LLMClient()
    engine = ReActEngine(llm_client=llm, registry=registry)

    if resume_from:
        checkpoint = checkpoint_mgr.load(resume_from)
        if checkpoint:
            engine.history = checkpoint.get("history", [])
            if RICH_AVAILABLE:
                console.print(f"[yellow]📂 从断点恢复: 步骤 {checkpoint['step']}[/yellow]")
            else:
                print(f"从断点恢复: 步骤 {checkpoint['step']}")
        else:
            if RICH_AVAILABLE:
                console.print("[red]断点加载失败，从头开始执行[/red]")
            else:
                print("断点加载失败，从头开始执行")

    # 记忆系统
    memory.start_task(task)

    # 事件发射
    bus.emit(EventType.TASK_START, {"task": task})

    if RICH_AVAILABLE:
        console.print(Panel(f"[bold]{task}[/bold]", title="🎯 任务", border_style="green"))
    else:
        print(f"任务: {task}")

    if plugin_count > 0:
        print(f"🔌 已加载 {plugin_count} 个插件")

    try:
        result = engine.run(task)
        if RICH_AVAILABLE:
            console.print(Panel(str(result), title="📋 执行结果", border_style="blue"))
        else:
            print(f"执行结果: {result}")

        # 记录经验
        memory.end_task(task, result[:200], success=True)
        bus.emit(EventType.TASK_END, {"task": task, "result": result[:100]})

    except KeyboardInterrupt:
        print("\n⚠️ 任务被用户中断")
        if ENABLE_CHECKPOINT:
            checkpoint_mgr.save(task=task, step=engine.step_count, history=engine.history)
            print("💾 断点已自动保存，可用 --resume 恢复")
        memory.end_task(task, "用户中断", success=False)
        bus.emit(EventType.TASK_ERROR, {"task": task, "error": "用户中断"})

    except Exception as e:
        logger.error(f"任务执行出错: {e}", exc_info=True)
        if RICH_AVAILABLE:
            console.print(f"[red]❌ 执行错误: {e}[/red]")
        else:
            print(f"执行错误: {e}")
        memory.end_task(task, str(e)[:200], success=False)
        bus.emit(EventType.TASK_ERROR, {"task": task, "error": str(e)})

    if guard.get_blocked_count() > 0:
        print(f"\n🛡️ 本次执行拦截了 {guard.get_blocked_count()} 个敏感操作")


def interactive_mode():
    """交互模式"""
    print_banner()

    from omnibot.decision import (
        LLMClient, create_builtin_registry, ReActEngine,
        ToolRegistry,
    )
    from omnibot.infrastructure import (
        Guardian, CheckpointManager, get_bus, MemorySystem,
        PluginManager, EventType,
    )

    bus = get_bus()
    guard = Guardian(event_bus=bus)
    memory = MemorySystem()
    registry = create_builtin_registry()

    plugin_mgr = PluginManager(registry)
    plugin_count = plugin_mgr.auto_load()

    llm = LLMClient()
    engine = ReActEngine(llm_client=llm, registry=registry)

    print("进入交互模式，输入任务指令，输入 quit/exit 退出\n")
    if plugin_count > 0:
        print(f"🔌 已加载 {plugin_count} 个插件")

    while True:
        try:
            task = input("OmniBot> ").strip()
            if not task:
                continue
            if task.lower() in ("quit", "exit", "q", "退出"):
                print("再见！")
                break

            if RICH_AVAILABLE:
                console.print(Panel(f"[bold]{task}[/bold]", title="🎯 任务", border_style="green"))
            else:
                print(f"任务: {task}")

            memory.start_task(task)
            bus.emit(EventType.TASK_START, {"task": task})

            try:
                result = engine.run(task)
                if RICH_AVAILABLE:
                    console.print(Panel(str(result), title="📋 执行结果", border_style="blue"))
                else:
                    print(f"执行结果: {result}")
                memory.end_task(task, result[:200], success=True)
            except KeyboardInterrupt:
                print("\n按 Ctrl+C 再次退出，或输入新任务继续")
            except Exception as e:
                logger.error(f"交互模式出错: {e}", exc_info=True)
                print(f"❌ 错误: {e}")
                memory.end_task(task, str(e)[:200], success=False)

        except KeyboardInterrupt:
            print("\n按 Ctrl+C 再次退出，或输入新任务继续")
        except EOFError:
            break


def list_tools():
    """列出所有可用工具"""
    from omnibot.decision import create_builtin_registry
    from omnibot.infrastructure import PluginManager

    registry = create_builtin_registry()
    plugin_mgr = PluginManager(registry)
    plugin_mgr.auto_load()

    print(f"\nOmniBot 2.0 可用工具 ({len(registry.list_tools())} 个):\n")
    for name in sorted(registry.list_tools()):
        tool = registry.get(name)
        print(f"  🔧 {name}: {tool.description}")
    print()


def list_checkpoints():
    """列出所有断点"""
    from omnibot.infrastructure import CheckpointManager
    mgr = CheckpointManager()
    checkpoints = mgr.list_all()
    if not checkpoints:
        print("暂无断点")
        return
    print(f"\n断点列表 ({len(checkpoints)} 个):\n")
    for cp in checkpoints:
        print(f"  📂 步骤{cp['step']}: {cp['task'][:50]}  ({cp['timestamp']})")
        print(f"     {cp['path']}")
    print()


def show_config():
    """显示当前配置"""
    from omnibot.infrastructure import dump
    config = dump()
    print("\n当前配置:\n")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print()


# ============================================================
# CLI 入口
# ============================================================

def main():
    _setup_logger()
    parser = argparse.ArgumentParser(
        description=f"OmniBot {__version__}（全能灵控）— 跨平台AI智能体",
    )
    parser.add_argument("--task", "-t", help="要执行的任务")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--resume", "-r", help="从断点恢复")
    parser.add_argument("--list-tools", action="store_true", help="列出可用工具")
    parser.add_argument("--list-checkpoints", action="store_true", help="列出断点")
    parser.add_argument("--config", action="store_true", help="显示配置")
    parser.add_argument("--version", "-v", action="version", version=f"OmniBot {__version__}")

    args = parser.parse_args()

    if args.config:
        show_config()
        return

    if args.list_tools:
        list_tools()
        return

    if args.list_checkpoints:
        list_checkpoints()
        return

    if args.interactive:
        interactive_mode()
    elif args.task:
        run_task(args.task, resume_from=args.resume)
    elif args.resume:
        run_task("", resume_from=args.resume)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
