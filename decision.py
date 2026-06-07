"""
OmniBot 2.0 决策层 — LLM 客户端 + 规划器 + ReAct 引擎 + 工具注册

职责：
- LLM 统一调用（OpenAI / Anthropic）
- Plan-and-Solve 规划
- ReAct 推理循环（思考-行动-观察）
- 工具注册表 + 10 个内置工具
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from omnibot.infrastructure import (
    ToolDef, ActionResult, ActionStatus,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
    LLM_BASE_URL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    PLAN_ENABLED, PLAN_MAX_STEPS,
    REACT_MAX_ITERATIONS, ACTION_MAX_RETRIES, ACTION_RETRY_DELAY,
)


# ============================================================
# LLM 客户端
# ============================================================

class LLMClient:
    """统一 LLM 调用客户端"""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None):
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None) -> dict[str, Any]:
        """调用 LLM，返回 {"content": str, "tool_calls": list | None}"""
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        if self.provider == "openai":
            return self._chat_openai(messages, tools, temperature)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, temperature)

    def _chat_openai(self, messages, tools, temperature):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        result = {"content": choice.message.content or "", "tool_calls": None}
        if choice.message.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }}
                for tc in choice.message.tool_calls
            ]
        return result

    def _chat_anthropic(self, messages, tools, temperature):
        system_msg, chat_messages = "", []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                chat_messages.append(m)

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg.strip()
        if tools:
            kwargs["tools"] = self._convert_tools_for_anthropic(tools)

        response = self._client.messages.create(**kwargs)
        result = {"content": "", "tool_calls": None}
        text_parts, tool_use_parts = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_parts.append({
                    "id": block.id,
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                })
        result["content"] = "\n".join(text_parts)
        if tool_use_parts:
            result["tool_calls"] = tool_use_parts
        return result

    @staticmethod
    def _convert_tools_for_anthropic(tools: list[dict]) -> list[dict]:
        """OpenAI 格式 → Anthropic 格式"""
        converted = []
        for t in tools:
            func = t.get("function", {})
            converted.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted

    def chat_with_vision(self, messages: list[dict], image_path: str,
                         temperature: float | None = None) -> dict[str, Any]:
        """视觉理解：分析截图"""
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        vision_messages = messages.copy()
        last_user = next((i for i, m in enumerate(vision_messages) if m["role"] == "user"), None)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"},
        }

        if last_user is not None:
            original = vision_messages[last_user]["content"]
            vision_messages[last_user]["content"] = [
                {"type": "text", "text": original if isinstance(original, str) else str(original)},
                image_content,
            ]
        else:
            vision_messages.append({"role": "user", "content": [image_content]})

        return self.chat(vision_messages, temperature=temperature)


# ============================================================
# 规划器
# ============================================================

@dataclass
class PlanStep:
    """规划步骤"""
    index: int
    description: str
    tool: str = ""                     # 建议使用的工具
    arguments: dict = field(default_factory=dict)
    done: bool = False
    result: str = ""


@dataclass
class Plan:
    """执行计划"""
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    raw_response: str = ""

    def current_step(self) -> PlanStep | None:
        """获取当前未完成的步骤"""
        for s in self.steps:
            if not s.done:
                return s
        return None

    def mark_done(self, index: int, result: str = ""):
        """标记步骤完成"""
        for s in self.steps:
            if s.index == index:
                s.done = True
                s.result = result
                break

    def progress(self) -> str:
        """进度摘要"""
        done = sum(1 for s in self.steps if s.done)
        total = len(self.steps)
        return f"[{done}/{total}]"

    def to_text(self) -> str:
        """转为文本描述"""
        lines = [f"任务: {self.task}", f"进度: {self.progress()}", ""]
        for s in self.steps:
            status = "✅" if s.done else "⏳"
            lines.append(f"  {status} 步骤{s.index}: {s.description}")
            if s.result:
                lines.append(f"      结果: {s.result[:80]}")
        return "\n".join(lines)


def create_plan(task: str, screenshot_path: str, llm_client) -> Plan:
    """创建执行计划

    Args:
        task: 用户任务描述
        screenshot_path: 当前屏幕截图路径
        llm_client: LLMClient 实例

    Returns:
        Plan 对象
    """
    if not PLAN_ENABLED:
        return Plan(task=task, steps=[PlanStep(index=1, description=task)])

    plan_prompt = (
        f"任务：{task}\n\n"
        f"请观察当前屏幕截图，列出完成此任务的具体步骤（最多{PLAN_MAX_STEPS}步）。\n\n"
        "要求：\n"
        "1. 每个步骤说清楚用什么工具、传什么参数\n"
        "2. 考虑可能的异常（元素找不到、页面没加载等）\n"
        "3. 步骤要具体可执行\n\n"
        "用JSON格式返回：\n"
        '{"steps": [{"index": 1, "description": "步骤描述", "tool": "工具名", "arguments": {参数}}]}\n'
        "只返回JSON。"
    )

    result = llm_client.chat_with_vision(
        messages=[{"role": "user", "content": plan_prompt}],
        image_path=screenshot_path,
        temperature=0.1,
    )

    raw = result.get("content", "")
    return _parse_plan(task, raw)


def _parse_plan(task: str, raw: str) -> Plan:
    """解析 LLM 返回的计划"""
    json_str = raw
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1:]:
            if part.startswith("json"):
                part = part[4:]
            json_str = part.strip()
            break

    try:
        data = json.loads(json_str.strip())
        steps_data = data.get("steps", [])
        steps = [
            PlanStep(
                index=s.get("index", i + 1),
                description=s.get("description", ""),
                tool=s.get("tool", ""),
                arguments=s.get("arguments", {}),
            )
            for i, s in enumerate(steps_data)
        ]
        return Plan(task=task, steps=steps, raw_response=raw)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Planner] 计划解析失败: {e}，降级为直接执行")
        return Plan(task=task, steps=[PlanStep(index=1, description=task)])


# ============================================================
# ReAct 引擎
# ============================================================

REACT_SYSTEM_PROMPT = """你是 OmniBot 2.0（全能灵控），一个能看懂屏幕并操控电脑的AI智能体。

你的工作方式是 Plan-and-Solve + ReAct：
1. **规划(Plan)**：先截图观察屏幕，列出完成任务的步骤
2. **执行(Solve)**：逐步执行，每步用 ReAct 循环（思考→行动→观察）
3. **验证(Verify)**：操作后用 frame_diff 或 screenshot 验证结果
4. **自愈(Heal)**：操作失败时自动重试，或尝试替代方案
5. 重复直到任务完成

关键原则：
- screenshot 使用LLM视觉读屏（不需要本地OCR），会返回屏幕文字描述
- 优先用 click_text 找字点击，不要用坐标点击
- 操作不确定时，用 frame_diff 观察屏幕变化来判断操作是否生效
- vision_find 用于定位无法用文字描述的元素（图标、图片按钮等）
- 等待界面加载（使用 wait）再继续
- 失败时分析原因，尝试替代方案（自愈）
- 使用中文回复

可用工具（按优先级）：
- screenshot: LLM视觉读屏（首选观察工具）
- click_text(text): 找字点击（首选操作工具）
- vision_find(description): 视觉定位（找图标等）
- frame_diff(wait_seconds?): 帧差分（检测操作是否生效）
- click(x, y): 坐标点击（备选）
- type_text(text, press_enter?): 输入文字
- hotkey(keys): 组合快捷键
- scroll(clicks): 滚动滚轮
- wait(seconds?): 等待
- save_screenshot(path): 保存截图
"""


class ReActEngine:
    """ReAct 推理引擎 2.0：思考-行动-观察 + 自愈重试"""

    def __init__(self, llm_client=None, registry: ToolRegistry | None = None):
        if llm_client is None:
            llm_client = LLMClient()
        self.llm = llm_client
        self.registry = registry or create_builtin_registry()
        self.step_count = 0
        self.history: list[dict] = []
        self._plan: Plan | None = None

    def _build_messages(self, task: str) -> list[dict]:
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"任务：{task}"},
        ]
        # 如果有计划，附加到 system
        if self._plan:
            messages[0]["content"] += f"\n\n当前执行计划:\n{self._plan.to_text()}"
        messages.extend(self.history)
        return messages

    def _execute_tool_with_retry(self, tool_name: str, arguments: dict) -> str:
        """带重试的工具执行"""
        last_error = ""
        for attempt in range(ACTION_MAX_RETRIES + 1):
            result = self.registry.execute(tool_name, arguments)

            if result.ok:
                return result.to_observation()

            last_error = result.error or "未知错误"
            if attempt < ACTION_MAX_RETRIES:
                print(f"[ReAct] 操作失败，重试 {attempt+1}/{ACTION_MAX_RETRIES}: {last_error}")
                time.sleep(ACTION_RETRY_DELAY)

        return f"操作失败（已重试{ACTION_MAX_RETRIES}次）: {last_error}"

    def run(self, task: str, max_iterations: int | None = None) -> str:
        """执行 Plan-and-Solve + ReAct 循环"""
        max_iter = max_iterations or REACT_MAX_ITERATIONS
        print(f"\n{'='*60}")
        print(f"🎯 任务: {task}")
        print(f"{'='*60}\n")

        # Phase 1: Plan
        print("📋 Phase 1: 规划中...")
        initial_path = take_screenshot(tag="initial")
        self._plan = create_plan(task, initial_path, self.llm)
        print(f"📋 执行计划:\n{self._plan.to_text()}\n")

        self.history.append({
            "role": "user",
            "content": f"执行计划:\n{self._plan.to_text()}\n\n请按照计划逐步执行任务：{task}",
        })

        # Phase 2: Solve
        print("🔧 Phase 2: 开始执行...")
        for iteration in range(max_iter):
            self.step_count = iteration + 1
            current = self._plan.current_step()
            step_desc = f"步骤 {current.index}: {current.description}" if current else f"第 {self.step_count} 步"
            print(f"\n--- {step_desc} / 最多 {max_iter} 步 ---")

            messages = self._build_messages(task)
            response = self.llm.chat(messages, tools=self.registry.to_openai_tools())

            thought = response.get("content", "")
            if thought:
                print(f"💭 思考: {thought[:200]}{'...' if len(thought) > 200 else ''}")

            tool_calls = response.get("tool_calls")
            if not tool_calls:
                self.history.append({"role": "assistant", "content": thought})
                # 标记当前计划步骤完成
                if current:
                    self._plan.mark_done(current.index, thought[:100])
                print(f"\n✅ 任务完成!\n结果: {thought}\n")
                return thought

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                print(f"🔧 执行: {func_name}({json.dumps(func_args, ensure_ascii=False)})")

                observation = self._execute_tool_with_retry(func_name, func_args)
                print(f"👁️ 观察: {observation[:200]}{'...' if len(observation) > 200 else ''}")

                self.history.append({
                    "role": "assistant",
                    "content": thought or "",
                    "tool_calls": [tc],
                })
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": observation,
                })

        print(f"\n⚠️ 达到最大迭代次数 ({max_iter})，任务可能未完全完成\n")
        return "达到最大迭代次数，任务未完成。"


# ============================================================
# 工具注册表
# ============================================================

class ToolRegistry:
    """工具注册表：统一管理所有工具的定义和执行"""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str, description: str,
                 parameters: dict | None = None,
                 handler: Callable | None = None):
        """注册一个工具

        Args:
            name: 工具名
            description: 工具描述
            parameters: OpenAI function calling 参数 schema
            handler: 处理函数 (arguments: dict) -> str
        """
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
        if handler:
            self._handlers[name] = handler

    def get(self, name: str) -> ToolDef | None:
        """获取工具定义"""
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict) -> ActionResult:
        """执行工具调用"""
        handler = self._handlers.get(name)
        if handler is None:
            return ActionResult(
                status=ActionStatus.FAILED,
                error=f"未知工具: {name}",
            )

        start = time.monotonic()
        try:
            result_data = handler(arguments)
            elapsed = (time.monotonic() - start) * 1000
            return ActionResult(
                status=ActionStatus.SUCCESS,
                data=result_data,
                elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return ActionResult(
                status=ActionStatus.FAILED,
                error=str(e),
                elapsed_ms=elapsed,
            )

    def to_openai_tools(self) -> list[dict]:
        """转为 OpenAI function calling 格式"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        """列出所有注册的工具名"""
        return list(self._tools.keys())

    def describe(self) -> str:
        """生成工具列表描述"""
        lines = [f"可用工具 ({len(self._tools)} 个):"]
        for name, tool in self._tools.items():
            lines.append(f"  - {name}: {tool.description}")
        return "\n".join(lines)


# ============================================================
# 内置工具注册
# ============================================================

def create_builtin_registry() -> ToolRegistry:
    """创建内置工具注册表"""
    registry = ToolRegistry()

    # --- screenshot: LLM 视觉读屏 ---
    registry.register(
        name="screenshot",
        description="截取当前屏幕，用LLM视觉读取屏幕内容（轻量！不需要本地OCR）",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_handle_screenshot,
    )

    # --- click_text: 找字点击 ---
    registry.register(
        name="click_text",
        description="在屏幕上查找文字并点击其中心位置（推荐！比坐标点击更可靠）",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要查找并点击的文字"},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "鼠标按键，默认left"},
                "clicks": {"type": "integer", "description": "点击次数，默认1"},
                "index": {"type": "integer", "description": "同屏多个匹配时选第几个，0=最匹配"},
            },
            "required": ["text"],
        },
        handler=_handle_click_text,
    )

    # --- vision_find: 视觉定位 ---
    registry.register(
        name="vision_find",
        description="用LLM视觉定位屏幕上的元素位置（适合找图标/图片按钮等非文字元素）",
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "元素描述，如'红色关闭按钮'"},
            },
            "required": ["description"],
        },
        handler=_handle_vision_find,
    )

    # --- frame_diff: 帧差分 ---
    registry.register(
        name="frame_diff",
        description="对比前后两帧屏幕，检测哪里发生了变化（轻量！用于判断操作是否生效、弹窗是否出现）",
        parameters={
            "type": "object",
            "properties": {
                "wait_seconds": {"type": "number", "description": "两帧之间等待秒数，默认0.5"},
            },
            "required": [],
        },
        handler=_handle_frame_diff,
    )

    # --- click: 坐标点击 ---
    registry.register(
        name="click",
        description="在指定坐标点击鼠标（仅当无法用文字描述目标时使用）",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X坐标"},
                "y": {"type": "integer", "description": "Y坐标"},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "鼠标按键"},
                "clicks": {"type": "integer", "description": "点击次数，默认1"},
            },
            "required": ["x", "y"],
        },
        handler=_handle_click,
    )

    # --- type_text: 输入文字 ---
    registry.register(
        name="type_text",
        description="在当前位置输入文字",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要输入的文字"},
                "press_enter": {"type": "boolean", "description": "输入后是否按回车"},
            },
            "required": ["text"],
        },
        handler=_handle_type_text,
    )

    # --- hotkey: 快捷键 ---
    registry.register(
        name="hotkey",
        description="按下组合快捷键（如 ctrl+v, alt+f4）",
        parameters={
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}, "description": "快捷键列表"},
            },
            "required": ["keys"],
        },
        handler=_handle_hotkey,
    )

    # --- wait: 等待 ---
    registry.register(
        name="wait",
        description="等待一段时间（用于等待页面加载等）",
        parameters={
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "等待秒数，默认2秒"},
            },
            "required": [],
        },
        handler=_handle_wait,
    )

    # --- save_screenshot: 保存截图 ---
    registry.register(
        name="save_screenshot",
        description="截屏并保存到指定路径",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "保存路径"},
            },
            "required": ["path"],
        },
        handler=_handle_save_screenshot,
    )

    # --- scroll: 滚动 ---
    registry.register(
        name="scroll",
        description="滚动鼠标滚轮",
        parameters={
            "type": "object",
            "properties": {
                "clicks": {"type": "integer", "description": "滚动量，正数向上，负数向下"},
                "x": {"type": "integer", "description": "X坐标（可选）"},
                "y": {"type": "integer", "description": "Y坐标（可选）"},
            },
            "required": ["clicks"],
        },
        handler=_handle_scroll,
    )

    return registry


# ============================================================
# 内置工具处理函数
# ============================================================

def _handle_screenshot(args: dict) -> str:
    from omnibot.infrastructure import take_screenshot
    from omnibot.infrastructure import vision_read
    path = take_screenshot(tag="step")
    vr = vision_read(path)
    return f"截图: {path}\n屏幕内容: {vr.description}"


def _handle_click_text(args: dict) -> str:
    from omnibot.executor import MouseController
    text = args["text"]
    button = args.get("button", "left")
    clicks = args.get("clicks", 1)
    index = args.get("index", 0)
    success = MouseController.click_text(text, button=button, clicks=clicks, index=index)
    if success:
        return f"已找到并点击了 \"{text}\""
    return f"未找到文字 \"{text}\"，尝试用 vision_find 定位，或用其他关键词"


def _handle_vision_find(args: dict) -> str:
    from omnibot.infrastructure import vision_find
    from omnibot.executor import MouseController
    desc = args["description"]
    coords = vision_find(desc)
    if coords:
        MouseController.click(x=coords[0], y=coords[1])
        return f"已通过视觉定位 \"{desc}\" 并点击 @ {coords}"
    return f"视觉定位未找到 \"{desc}\"，请截图观察后重试"


def _handle_frame_diff(args: dict) -> str:
    from omnibot.infrastructure import quick_diff
    wait_sec = args.get("wait_seconds", 0.5)
    changes = quick_diff(wait_seconds=wait_sec)
    if not changes:
        return "屏幕无明显变化（操作可能未生效，或已加载完毕）"
    lines = [f"检测到 {len(changes)} 个变化区域:"]
    for i, c in enumerate(changes[:5]):
        lines.append(f"  区域{i+1}: 中心({c.center[0]},{c.center[1]}) "
                     f"范围{c.bbox} 变化{c.change_ratio*100:.1f}%")
    return "\n".join(lines)


def _handle_click(args: dict) -> str:
    from omnibot.executor import MouseController
    x, y = args["x"], args["y"]
    button = args.get("button", "left")
    clicks = args.get("clicks", 1)
    MouseController.click(x=x, y=y, button=button, clicks=clicks)
    return f"已在 ({x}, {y}) 执行 {button} 点击 ×{clicks}"


def _handle_type_text(args: dict) -> str:
    from omnibot.executor import KeyboardController
    text = args["text"]
    press_enter = args.get("press_enter", False)
    KeyboardController.type_text(text)
    if press_enter:
        KeyboardController.press("enter")
    return f"已输入文字: {text}" + (" + 回车" if press_enter else "")


def _handle_hotkey(args: dict) -> str:
    from omnibot.executor import KeyboardController
    keys = args["keys"]
    KeyboardController.hotkey(*keys)
    return f"已按下快捷键: {'+'.join(keys)}"


def _handle_wait(args: dict) -> str:
    seconds = args.get("seconds", 2)
    time.sleep(seconds)
    return f"已等待 {seconds} 秒"


def _handle_save_screenshot(args: dict) -> str:
    import shutil
    from omnibot.infrastructure import take_screenshot
    path = args["path"]
    screenshot_path = take_screenshot(tag="save")
    shutil.copy2(screenshot_path, path)
    return f"截图已保存到: {path}"


def _handle_scroll(args: dict) -> str:
    from omnibot.executor import MouseController
    clicks = args["clicks"]
    x = args.get("x")
    y = args.get("y")
    MouseController.scroll(clicks, x=x, y=y)
    direction = "上" if clicks > 0 else "下"
    return f"已向{direction}滚动 {abs(clicks)} 格"
