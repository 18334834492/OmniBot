"""
OmniBot 2.0 基础设施层 — 协议 + 配置 + 事件 + 插件 + 记忆 + 安全

职责：
- 核心协议定义（Action、Event、Capability、Memory）
- 配置中心（环境变量、运行时覆盖）
- 事件总线（发布-订阅模式）
- 插件系统（能力注册、动态加载）
- 记忆系统（短期 + 长期）
- 安全守护（敏感操作拦截、授权）
"""

from __future__ import annotations

import json
import os
import time
import logging
import platform
import importlib
import importlib.util
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from PIL import Image, ImageGrab


# ============================================================
# 配置中心
# ============================================================

def _load_dotenv():
    """尝试加载 .env 文件（零依赖，纯 Python 解析）"""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("\"'")
            if key and key not in os.environ:  # 环境变量优先
                os.environ[key] = val


_load_dotenv()


# 项目基础
PROJECT_NAME = "OmniBot"
VERSION = "2.0.0"
PROJECT_DIR = Path(__file__).parent.parent
SCREENSHOT_DIR = PROJECT_DIR / "screenshots"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "data"

# 感知层
PERCEPTION_MODE = "llm_vision"      # llm_vision / ocr / hybrid
SCREENSHOT_SCALE = 1.0
SCREENSHOT_QUALITY = 85             # JPEG 质量（仅缩略图用）

# 帧差分
FRAME_DIFF_THRESHOLD = 30           # 像素差异阈值
FRAME_DIFF_MIN_AREA = 200           # 最小变化面积
FRAME_DIFF_WAIT = 0.5               # 两帧间隔（秒）

# LLM 视觉
VISION_DETAIL = "high"              # low / high / auto
VISION_MAX_RETRIES = 2              # 视觉调用重试

# 决策层
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # openai / anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")          # 自定义端点
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.1
REACT_MAX_ITERATIONS = 15           # 2.0 增加到 15 步

# Plan-and-Solve
PLAN_ENABLED = True                 # 是否启用规划阶段
PLAN_MAX_STEPS = 8                  # 规划最大步骤数

# 执行层
PLATFORM = platform.system()
MOUSE_MOVE_DURATION = 0.3
KEYPRESS_INTERVAL = 0.05
ACTION_MAX_RETRIES = 2              # 操作重试次数
ACTION_RETRY_DELAY = 1.0            # 重试间隔（秒）
ACTION_SCREENSHOT = True            # 操作前后自动截图

# 安全边界
SAFE_MODE = True
SENSITIVE_OPERATIONS = [
    "delete", "remove", "format",
    "shutdown", "restart",
    "send_email", "transfer_money",
    "modify_system", "install",
]
REQUIRE_CONFIRMATION = True
AUTO_SCREENSHOT_BEFORE_OP = True
GUARD_AUTO_ROLLBACK = False         # 失败后自动回滚（实验性）

# 断点续做
ENABLE_CHECKPOINT = True
CHECKPOINT_INTERVAL = 3

# 记忆系统
MEMORY_ENABLED = True
MEMORY_SHORT_LIMIT = 50             # 短期记忆条数上限
MEMORY_LONG_LIMIT = 500             # 长期经验条数上限
MEMORY_SAVE_INTERVAL = 5            # 每N步自动保存

# 插件系统
PLUGIN_DIR = PROJECT_DIR / "plugins"
PLUGIN_AUTO_LOAD = True             # 启动时自动加载插件

# 运行时覆盖
_overrides: dict[str, Any] = {}


def get(key: str, default: Any = None) -> Any:
    """获取配置值（支持运行时覆盖）"""
    if key in _overrides:
        return _overrides[key]
    return globals().get(key, default)


def set(key: str, value: Any) -> None:
    """运行时覆盖配置值"""
    _overrides[key] = value


def reset(key: str | None = None) -> None:
    """重置覆盖（None = 全部重置）"""
    if key is None:
        _overrides.clear()
    else:
        _overrides.pop(key, None)


def dump() -> dict[str, Any]:
    """导出所有配置（排除敏感信息）"""
    sensitive = {"LLM_API_KEY"}
    result = {}
    for k, v in globals().items():
        if k.startswith("_") or k in sensitive or callable(v):
            continue
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
    result.update(_overrides)
    return result


# ============================================================
# 核心协议
# ============================================================

# Action 协议 — 统一执行层操作
class ActionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class ActionResult:
    """操作执行结果"""
    status: ActionStatus = ActionStatus.PENDING
    data: Any = None
    error: str | None = None
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    retry_count: int = 0
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    def to_observation(self) -> str:
        """转为 LLM 可读的观察文本"""
        if self.ok:
            return str(self.data) if self.data else "操作成功"
        return f"操作失败: {self.error}" if self.error else "操作失败"


@dataclass
class Action:
    """待执行的操作"""
    name: str                           # 工具名
    arguments: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2                # 最大重试次数
    retry_delay: float = 1.0            # 重试间隔（秒）
    require_screenshot: bool = True     # 操作前后是否截图
    rollback_fn: Callable | None = None # 回滚函数（可选）


# Event 协议 — 事件总线
class EventType(str, Enum):
    # 生命周期
    TASK_START = "task.start"
    TASK_END = "task.end"
    TASK_ERROR = "task.error"

    # 计划
    PLAN_CREATED = "plan.created"
    PLAN_STEP_START = "plan.step.start"
    PLAN_STEP_END = "plan.step.end"

    # 操作
    ACTION_BEFORE = "action.before"
    ACTION_AFTER = "action.after"
    ACTION_FAILED = "action.failed"
    ACTION_RETRY = "action.retry"
    ACTION_ROLLBACK = "action.rollback"

    # 感知
    SCREENSHOT_TAKEN = "perception.screenshot"
    VISION_READ = "perception.vision_read"
    FRAME_DIFF = "perception.frame_diff"

    # 安全
    GUARD_BLOCKED = "guard.blocked"
    GUARD_ALLOWED = "guard.allowed"


@dataclass
class Event:
    """事件"""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0  # 自动填充


# Capability 协议 — 插件能力注册
@dataclass
class ToolDef:
    """工具定义（注册到 LLM 的 function calling schema）"""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable | None = None     # 绑定的处理函数

    def to_openai_schema(self) -> dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class Capability:
    """插件声明的能力"""
    name: str                           # 能力名（如 "browser", "file_ops"）
    description: str = ""
    tools: list[ToolDef] = field(default_factory=list)
    version: str = "1.0.0"


# Memory 协议 — 记忆系统
@dataclass
class MemoryEntry:
    """记忆条目"""
    key: str
    value: Any
    source: str = ""        # 来源（如 "vision", "user", "experience"）
    timestamp: float = 0.0
    access_count: int = 0


# ============================================================
# 事件总线
# ============================================================

logger = logging.getLogger("OmniBot.EventBus")


class EventBus:
    """事件总线：发布-订阅模式"""

    def __init__(self):
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)
        self._wildcard_subscribers: list[Callable] = []    # 订阅所有事件
        self._interceptors: list[Callable] = []            # 操作前拦截器
        self._history: list[Event] = []
        self._history_limit = 1000

    # ============================================================
    # 订阅
    # ============================================================

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """订阅指定事件类型"""
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: Callable) -> None:
        """订阅所有事件（通配符）"""
        self._wildcard_subscribers.append(handler)

    def add_interceptor(self, interceptor: Callable) -> None:
        """添加操作前拦截器

        拦截器签名: (event: Event) -> bool
        返回 False 可阻止操作
        """
        self._interceptors.append(interceptor)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """取消订阅"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    # ============================================================
    # 发布
    # ============================================================

    def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        """发布事件

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            发布的事件对象
        """
        event = Event(
            type=event_type,
            data=data or {},
            timestamp=time.time(),
        )

        # 记录历史
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        # 通知订阅者
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"事件处理器错误 [{event_type}]: {e}")

        # 通知通配符订阅者
        for handler in self._wildcard_subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"通配符处理器错误 [{event_type}]: {e}")

        return event

    def emit_and_check(self, event_type: EventType,
                       data: dict[str, Any] | None = None) -> bool:
        """发布事件并检查拦截器（用于操作前检查）

        Returns:
            True = 允许操作，False = 被拦截
        """
        event = Event(type=event_type, data=data or {}, timestamp=time.time())

        for interceptor in self._interceptors:
            try:
                if not interceptor(event):
                    logger.info(f"操作被拦截器阻止: {event_type}")
                    return False
            except Exception as e:
                logger.error(f"拦截器错误: {e}")

        # 通过拦截器，正常发布
        self.emit(event_type, data)
        return True

    # ============================================================
    # 查询
    # ============================================================

    def get_history(self, event_type: EventType | None = None,
                    limit: int = 50) -> list[Event]:
        """获取事件历史"""
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """清空事件历史"""
        self._history.clear()

    def stats(self) -> dict[str, int]:
        """事件统计"""
        counts: dict[str, int] = defaultdict(int)
        for e in self._history:
            counts[e.type.value] += 1
        return dict(counts)


# 全局单例
_global_bus: EventBus | None = None


def get_bus() -> EventBus:
    """获取全局事件总线"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_bus() -> None:
    """重置全局事件总线（测试用）"""
    global _global_bus
    _global_bus = None


# ============================================================
# 插件系统
# ============================================================

logger = logging.getLogger("OmniBot.Plugin")


class PluginInterface:
    """插件接口：所有插件必须继承此类"""

    # 插件元信息
    name: str = "unnamed"
    version: str = "1.0.0"
    description: str = ""

    def on_load(self, registry: ToolRegistry) -> Capability | None:
        """加载时调用，注册工具和声明能力

        Args:
            registry: 工具注册表

        Returns:
            Capability 对象（可选）
        """
        return None

    def on_unload(self) -> None:
        """卸载时调用（清理资源）"""
        pass

    def on_event(self, event_type: str, data: dict) -> None:
        """事件通知（可选）"""
        pass


class PluginManager:
    """插件管理器"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._plugins: dict[str, PluginInterface] = {}
        self._capabilities: dict[str, Capability] = {}

    def load_plugin(self, plugin: PluginInterface) -> bool:
        """加载一个插件实例"""
        try:
            capability = plugin.on_load(self.registry)
            self._plugins[plugin.name] = plugin
            if capability:
                self._capabilities[plugin.name] = capability
                logger.info(f"插件已加载: {plugin.name} v{plugin.version} — {plugin.description}")
                # 将插件的工具注册到工具注册表
                for tool in capability.tools:
                    if tool.handler:
                        self.registry.register(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                            handler=tool.handler,
                        )
                        logger.info(f"  注册工具: {tool.name}")
            else:
                logger.info(f"插件已加载: {plugin.name} v{plugin.version}")
            return True
        except Exception as e:
            logger.error(f"插件加载失败 [{plugin.name}]: {e}")
            return False

    def unload_plugin(self, name: str) -> bool:
        """卸载插件"""
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        try:
            plugin.on_unload()
            del self._plugins[name]
            self._capabilities.pop(name, None)
            logger.info(f"插件已卸载: {name}")
            return True
        except Exception as e:
            logger.error(f"插件卸载失败 [{name}]: {e}")
            return False

    def load_from_file(self, filepath: str | Path) -> bool:
        """从 Python 文件加载插件

        文件中必须定义一个 Plugin 类（继承 PluginInterface）
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"插件文件不存在: {filepath}")
            return False

        module_name = f"omnibot_plugin_{filepath.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                logger.error(f"无法加载模块: {filepath}")
                return False
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 查找 Plugin 类
            plugin_class = getattr(module, "Plugin", None)
            if plugin_class is None:
                # 尝试查找任何继承 PluginInterface 的类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, PluginInterface) and
                        attr is not PluginInterface):
                        plugin_class = attr
                        break

            if plugin_class is None:
                logger.error(f"插件文件中未找到 Plugin 类: {filepath}")
                return False

            plugin_instance = plugin_class()
            return self.load_plugin(plugin_instance)
        except Exception as e:
            logger.error(f"从文件加载插件失败 [{filepath}]: {e}")
            return False

    def auto_load(self) -> int:
        """自动加载 plugins/ 目录下的所有插件

        Returns:
            成功加载的插件数
        """
        if not PLUGIN_AUTO_LOAD:
            return 0

        count = 0
        plugin_dir = PLUGIN_DIR
        if not plugin_dir.exists():
            plugin_dir.mkdir(parents=True, exist_ok=True)
            # 创建示例插件
            self._create_example_plugin(plugin_dir)
            return 0

        for filepath in sorted(plugin_dir.glob("*.py")):
            if filepath.name.startswith("_"):
                continue
            if self.load_from_file(filepath):
                count += 1

        logger.info(f"\n已加载 {count} 个插件")
        return count

    def _create_example_plugin(self, plugin_dir: Path) -> None:
        """创建示例插件"""
        example = plugin_dir / "_example_plugin.py"
        example.write_text("""
# 示例插件：浏览器操作
from omnibot.infrastructure import PluginInterface, ToolDef

class Plugin(PluginInterface):
    name = "browser"
    version = "1.0.0"
    description = "浏览器操作能力"

    def on_load(self, registry):
        return Capability(
            name="browser",
            description="浏览器操作能力",
            tools=[
                ToolDef(
                    name="open_browser",
                    description="打开浏览器",
                    parameters={
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                    handler=lambda args: "浏览器已打开",
                ),
            ],
        )
""")
        logger.info(f"已创建示例插件: {example}")


# ============================================================
# 记忆系统
# ============================================================

logger = logging.getLogger("OmniBot.Memory")


class ShortTermMemory:
    """短期记忆：当前任务上下文"""

    def __init__(self, limit: int | None = None):
        self.limit = limit or MEMORY_SHORT_LIMIT
        self._entries: list[MemoryEntry] = []
        self._summary: str = ""

    def add(self, key: str, value, source: str = "") -> None:
        """添加短期记忆"""
        entry = MemoryEntry(
            key=key,
            value=value,
            source=source,
            timestamp=time.time(),
        )
        self._entries.append(entry)

        # 超限时压缩
        if len(self._entries) > self.limit:
            self._compress()

    def get(self, key: str, default=None):
        """获取最近一条匹配的记忆"""
        for entry in reversed(self._entries):
            if entry.key == key:
                entry.access_count += 1
                return entry.value
        return default

    def get_all(self, key: str | None = None) -> list[MemoryEntry]:
        """获取所有匹配的记忆"""
        if key is None:
            return list(self._entries)
        return [e for e in self._entries if e.key == key]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        """获取最近 N 条记忆"""
        return self._entries[-n:]

    def _compress(self) -> None:
        """压缩短期记忆：保留最近的一半 + 摘要"""
        keep_count = self.limit // 2
        removed = self._entries[:-keep_count]

        # 生成摘要
        if removed:
            summary_parts = []
            for e in removed[-5:]:  # 只取最近5条
                summary_parts.append(f"{e.key}: {str(e.value)[:50]}")
            self._summary += "\n" + ";".join(summary_parts) if summary_parts else ""

        self._entries = self._entries[-keep_count:]

    @property
    def summary(self) -> str:
        """获取记忆摘要"""
        parts = []
        if self._summary:
            parts.append(f"[历史摘要] {self._summary[:200]}")
        recent = self.recent(5)
        if recent:
            parts.append("[最近记忆]")
            for e in recent:
                parts.append(f"  {e.key}: {str(e.value)[:80]}")
        return "\n".join(parts)

    def clear(self) -> None:
        """清空短期记忆"""
        self._entries.clear()
        self._summary = ""

    def to_context(self) -> str:
        """转为 LLM 上下文"""
        if not self._entries and not self._summary:
            return ""
        return f"当前记忆:\n{self.summary}"

    def size(self) -> int:
        return len(self._entries)


class LongTermMemory:
    """长期经验库：跨任务可复用经验"""

    def __init__(self, limit: int | None = None):
        self.limit = limit or MEMORY_LONG_LIMIT
        self._experiences: list[dict] = []
        self._file = DATA_DIR / "experience.json"

    def add_experience(self, task: str, approach: str,
                       success: bool, notes: str = "") -> None:
        """记录一次任务经验"""
        self._experiences.append({
            "task": task,
            "approach": approach,
            "success": success,
            "notes": notes,
            "timestamp": time.time(),
        })

        # 超限淘汰最旧的失败经验
        if len(self._experiences) > self.limit:
            # 优先保留成功经验
            self._experiences.sort(key=lambda x: (not x["success"], x["timestamp"]))
            self._experiences = self._experiences[-self.limit:]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """搜索相关经验（简单关键词匹配）"""
        query_lower = query.lower()
        scored = []
        for exp in self._experiences:
            # 关键词匹配打分
            score = 0
            task_lower = exp["task"].lower()
            if query_lower in task_lower:
                score += 10
            # 逐词匹配
            for word in query_lower.split():
                if word in task_lower:
                    score += 3
                if word in exp.get("approach", "").lower():
                    score += 2
            if score > 0:
                scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:limit]]

    def get_relevant_tips(self, task: str) -> str:
        """获取与任务相关的经验提示"""
        results = self.search(task, limit=3)
        if not results:
            return ""

        lines = ["相关经验:"]
        for exp in results:
            status = "✅成功" if exp["success"] else "❌失败"
            lines.append(f"  {status} {exp['task']} → {exp['approach'][:60]}")
            if exp.get("notes"):
                lines.append(f"    备注: {exp['notes'][:80]}")
        return "\n".join(lines)

    def save(self) -> None:
        """持久化到文件"""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._experiences, ensure_ascii=False, indent=2))

    def load(self) -> None:
        """从文件加载"""
        if not self._file.exists():
            return
        try:
            self._experiences = json.loads(self._file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"加载长期记忆失败: {e}")


# ============================================================
# 安全守护
# ============================================================

logger = logging.getLogger("OmniBot.Guardian")


class Guardian:
    """安全守护 2.0：敏感操作识别 + 授权 + 回滚 + 事件发射"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.safe_mode = SAFE_MODE
        self.require_confirmation = REQUIRE_CONFIRMATION
        self.auto_screenshot = AUTO_SCREENSHOT_BEFORE_OP
        self.auto_rollback = GUARD_AUTO_ROLLBACK
        self.operation_log: list[dict] = []
        self._snapshots: dict[str, str] = {}  # 操作前截图缓存

    def check_operation(self, tool_name: str, arguments: dict) -> dict:
        """检查操作是否安全

        Returns:
            {"allowed": bool, "reason": str, "screenshot": str|None}
        """
        if not self.safe_mode:
            return {"allowed": True, "reason": "安全模式已关闭", "screenshot": None}

        screenshot_path = None
        if self.auto_screenshot:
            try:
                screenshot_path = take_screenshot(tag="pre_op")
            except Exception:
                logger.debug("操作前截图失败（无桌面环境），跳过")

        if self._is_sensitive_operation(tool_name, arguments):
            # 发射安全事件
            if self.event_bus:
                bus = self.event_bus
                bus.emit(EventType.GUARD_BLOCKED, {
                    "tool": tool_name, "arguments": arguments,
                })

            if self.require_confirmation:
                if not self._request_confirmation(tool_name, arguments, screenshot_path):
                    self._log_operation(tool_name, arguments, blocked=True)
                    return {
                        "allowed": False,
                        "reason": f"敏感操作被拒绝: {tool_name}",
                        "screenshot": screenshot_path,
                    }

        # 通过检查
        if self.event_bus:
            self.event_bus.emit(EventType.GUARD_ALLOWED, {
                "tool": tool_name, "arguments": arguments,
            })

        self._log_operation(tool_name, arguments, blocked=False)

        # 缓存操作前截图（用于回滚参考）
        if screenshot_path:
            self._snapshots[tool_name] = screenshot_path

        return {"allowed": True, "reason": "操作已授权", "screenshot": screenshot_path}

    def _is_sensitive_operation(self, tool_name: str, arguments: dict) -> bool:
        """判断是否为敏感操作"""
        operation_text = f"{tool_name} {json.dumps(arguments, ensure_ascii=False)}".lower()
        for keyword in SENSITIVE_OPERATIONS:
            if keyword in operation_text:
                return True
        if tool_name == "type_text":
            text = arguments.get("text", "").lower()
            if any(st in text for st in ["密码", "password", "passwd", "pin"]):
                return True
        return False

    def _request_confirmation(self, tool_name: str, arguments: dict,
                              screenshot_path: str | None) -> bool:
        """请求用户确认"""
        print("\n" + "⚠️ " * 10)
        print("🚨 检测到敏感操作，需要确认！")
        print(f"   操作: {tool_name}")
        print(f"   参数: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        if screenshot_path:
            print(f"   操作前截图: {screenshot_path}")
        print("⚠️ " * 10)
        try:
            return input("是否允许此操作？(y/N): ").strip().lower() in ("y", "yes", "是")
        except (EOFError, KeyboardInterrupt):
            print("\n操作已取消")
            return False

    def _log_operation(self, tool_name: str, arguments: dict, blocked: bool):
        """记录操作日志"""
        self.operation_log.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "arguments": arguments,
            "blocked": blocked,
        })

    def rollback(self, tool_name: str) -> bool:
        """尝试回滚操作（实验性）

        目前仅支持截图对比，无法真正撤销操作
        """
        snapshot = self._snapshots.get(tool_name)
        if not snapshot:
            logger.warning(f"无可用的回滚快照: {tool_name}")
            return False

        logger.info(f"回滚参考 - 操作前截图: {snapshot}")
        # 截一张当前状态
        current = take_screenshot(tag="rollback_current")

        if self.event_bus:
            self.event_bus.emit(EventType.ACTION_ROLLBACK, {
                "tool": tool_name,
                "before": snapshot,
                "after": current,
            })

        return True

    def get_log(self) -> list[dict]:
        return self.operation_log

    def get_blocked_count(self) -> int:
        return sum(1 for op in self.operation_log if op["blocked"])
