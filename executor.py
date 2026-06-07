"""
OmniBot 2.0 执行层 — 鼠标/键盘控制 + 找字点击

特性：
- pyautogui 延迟导入（无桌面环境不报错）
- 找字点击升级为 LLM 视觉辅助
- 跨平台剪贴板输入
"""
from __future__ import annotations

import subprocess

from omnibot.infrastructure import PLATFORM, MOUSE_MOVE_DURATION, KEYPRESS_INTERVAL

# pyautogui 延迟导入
_pyautogui = None


def _get_pyautogui():
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        _pyautogui = pyautogui
    return _pyautogui


# ============================================================
# 鼠标控制
# ============================================================

class MouseController:
    """鼠标控制器"""

    @staticmethod
    def move_to(x: int, y: int, duration: float | None = None):
        _get_pyautogui().moveTo(x=x, y=y, duration=duration or MOUSE_MOVE_DURATION)

    @staticmethod
    def click(x: int | None = None, y: int | None = None,
              button: str = "left", clicks: int = 1, duration: float | None = None):
        pag = _get_pyautogui()
        if x is not None and y is not None:
            pag.click(x=x, y=y, button=button, clicks=clicks,
                      duration=duration or MOUSE_MOVE_DURATION)
        else:
            pag.click(button=button, clicks=clicks)

    @staticmethod
    def double_click(x: int | None = None, y: int | None = None):
        pag = _get_pyautogui()
        pag.doubleClick(x=x, y=y) if x is not None else pag.doubleClick()

    @staticmethod
    def right_click(x: int | None = None, y: int | None = None):
        pag = _get_pyautogui()
        pag.rightClick(x=x, y=y) if x is not None else pag.rightClick()

    @staticmethod
    def drag_to(start_x: int, start_y: int, end_x: int, end_y: int,
                duration: float = 0.5, button: str = "left"):
        pag = _get_pyautogui()
        pag.moveTo(start_x, start_y)
        pag.drag(end_x - start_x, end_y - start_y, duration=duration, button=button)

    @staticmethod
    def scroll(clicks: int, x: int | None = None, y: int | None = None):
        _get_pyautogui().scroll(clicks, x=x, y=y)

    @staticmethod
    def get_position() -> tuple[int, int]:
        return _get_pyautogui().position()

    @staticmethod
    def click_text(text: str, button: str = "left", clicks: int = 1,
                   index: int = 0, duration: float | None = None) -> bool:
        """找字点击：LLM 视觉定位 → 点击

        Args:
            text: 要点击的文字
            button: 鼠标按键
            clicks: 点击次数
            index: 同屏多个匹配时选第几个
            duration: 移动动画时长

        Returns:
            True=成功点击，False=未找到
        """
        from omnibot.infrastructure import take_screenshot
        from omnibot.infrastructure import vision_read, vision_find

        # 先尝试 vision_read 找文字
        screenshot_path = take_screenshot(save=True, tag="click_text")
        vr = vision_read(screenshot_path)
        elem = vr.find_element(text)

        if elem and "x" in elem and "y" in elem:
            from omnibot.infrastructure import get_screen_size
            sw, sh = get_screen_size()
            px = int(elem["x"] / 100 * sw)
            py = int(elem["y"] / 100 * sh)
            MouseController.click(x=px, y=py, button=button, clicks=clicks, duration=duration)
            print(f"[执行] 🖱️ 已点击 \"{text}\" @ ({px}, {py})")
            return True

        # 降级到 vision_find
        coords = vision_find(text, screenshot_path)
        if coords:
            MouseController.click(x=coords[0], y=coords[1], button=button,
                                  clicks=clicks, duration=duration)
            print(f"[执行] 🖱️ 已点击 \"{text}\" @ {coords}")
            return True

        print(f"[执行] ❌ 未找到 \"{text}\"")
        return False


# ============================================================
# 键盘控制
# ============================================================

class KeyboardController:
    """键盘控制器"""

    SPECIAL_KEYS = {
        "enter": "enter", "return": "return", "tab": "tab",
        "escape": "escape", "esc": "escape", "backspace": "backspace",
        "delete": "delete", "space": "space",
        "up": "up", "down": "down", "left": "left", "right": "right",
        "home": "home", "end": "end", "pageup": "pageup", "pagedown": "pagedown",
        "capslock": "capslock", "shift": "shift", "ctrl": "ctrl",
        "alt": "alt", "cmd": "command", "win": "win",
        **{f"f{i}": f"f{i}" for i in range(1, 13)},
    }

    @staticmethod
    def type_text(text: str, interval: float | None = None):
        pag = _get_pyautogui()
        interval = interval or KEYPRESS_INTERVAL
        if text.isascii():
            pag.typewrite(text, interval=interval)
        else:
            KeyboardController._type_via_clipboard(text)

    @staticmethod
    def _type_via_clipboard(text: str):
        pag = _get_pyautogui()
        if PLATFORM == "Darwin":
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(text.encode("utf-8"))
            pag.hotkey("command", "v")
        elif PLATFORM == "Windows":
            p = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
            p.communicate(text.encode("gbk", errors="replace"))
            pag.hotkey("ctrl", "v")
        elif PLATFORM == "Linux":
            p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            p.communicate(text.encode("utf-8"))
            pag.hotkey("ctrl", "v")

    @staticmethod
    def press(key: str):
        _get_pyautogui().press(KeyboardController.SPECIAL_KEYS.get(key.lower(), key))

    @staticmethod
    def hotkey(*keys: str):
        mapped = [KeyboardController.SPECIAL_KEYS.get(k.lower(), k) for k in keys]
        _get_pyautogui().hotkey(*mapped)

    @staticmethod
    def key_down(key: str):
        _get_pyautogui().keyDown(KeyboardController.SPECIAL_KEYS.get(key.lower(), key))

    @staticmethod
    def key_up(key: str):
        _get_pyautogui().keyUp(KeyboardController.SPECIAL_KEYS.get(key.lower(), key))
