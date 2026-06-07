"""
OmniBot 2.0 感知层 — 截图 + LLM 视觉读屏 + 帧差分

职责：
- 截取屏幕/区域、管理截图文件、压缩图片
- 用 LLM 视觉能力理解屏幕内容、定位元素
- 对比前后帧，检测屏幕变化区域
"""

from __future__ import annotations

import os
import time
import base64
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageGrab

from omnibot.infrastructure import (
    SCREENSHOT_DIR, SCREENSHOT_SCALE,
    VISION_DETAIL, VISION_MAX_RETRIES,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS,
    FRAME_DIFF_THRESHOLD, FRAME_DIFF_MIN_AREA, FRAME_DIFF_WAIT,
)


# ============================================================
# 截图模块
# ============================================================

def ensure_dir() -> None:
    """确保截图目录存在"""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def take_screenshot(save: bool = True, tag: str = "") -> str:
    """截取当前屏幕并保存，返回文件路径"""
    ensure_dir()
    img = ImageGrab.grab()

    if SCREENSHOT_SCALE != 1.0:
        new_size = (int(img.width * SCREENSHOT_SCALE), int(img.height * SCREENSHOT_SCALE))
        img = img.resize(new_size, Image.LANCZOS)

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag_suffix = f"_{tag}" if tag else ""
        filename = f"screenshot_{timestamp}{tag_suffix}.png"
        filepath = str(SCREENSHOT_DIR / filename)
        img.save(filepath)
        return filepath

    temp_path = str(SCREENSHOT_DIR / "_temp_latest.png")
    img.save(temp_path)
    return temp_path


def take_region_screenshot(region: tuple[int, int, int, int],
                           save: bool = True, tag: str = "") -> str:
    """截取指定区域，返回文件路径"""
    ensure_dir()
    img = ImageGrab.grab(bbox=region)

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag_suffix = f"_{tag}" if tag else ""
        filename = f"region_{timestamp}{tag_suffix}.png"
        filepath = str(SCREENSHOT_DIR / filename)
        img.save(filepath)
        return filepath

    temp_path = str(SCREENSHOT_DIR / "_temp_region.png")
    img.save(temp_path)
    return temp_path


def get_screen_size() -> tuple[int, int]:
    """获取屏幕分辨率"""
    img = ImageGrab.grab()
    return img.size


def compress_for_vision(image_path: str, max_dim: int = 1280, quality: int = 85) -> str:
    """压缩图片用于 LLM 视觉调用（减少 token 消耗）

    Args:
        image_path: 原始截图路径
        max_dim: 最大边长（超过则缩放）
        quality: JPEG 质量

    Returns:
        压缩后的文件路径
    """
    img = Image.open(image_path)
    w, h = img.size

    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    out_path = image_path.rsplit(".", 1)[0] + "_compressed.jpg"
    img.convert("RGB").save(out_path, "JPEG", quality=quality)
    return out_path


# ============================================================
# LLM 视觉读屏
# ============================================================

@dataclass
class VisionResult:
    """LLM 视觉读屏结果"""
    description: str = ""                       # 屏幕内容描述
    elements: list[dict[str, Any]] = field(default_factory=list)  # 可交互元素
    raw_response: str = ""                      # 原始响应

    def find_element(self, text: str) -> dict | None:
        """在识别到的元素中查找包含指定文字的元素"""
        text_lower = text.lower()
        for elem in self.elements:
            elem_text = elem.get("text", "").lower()
            if text_lower in elem_text:
                return elem
        return None

    def to_observation(self) -> str:
        """转为 LLM 可读的观察文本"""
        lines = [f"屏幕内容: {self.description}"]
        if self.elements:
            lines.append(f"可交互元素 ({len(self.elements)} 个):")
            for e in self.elements[:10]:
                lines.append(f"  - {e.get('text', '?')} @ ({e.get('x', '?')}%, {e.get('y', '?')}%)")
        return "\n".join(lines)


def _encode_image(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def vision_read(image_path: str, prompt: str | None = None,
                llm_client=None) -> VisionResult:
    """用 LLM 视觉能力读取屏幕内容

    Args:
        image_path: 截图文件路径
        prompt: 自定义读屏指令
        llm_client: LLMClient 实例（不传则自动创建）

    Returns:
        VisionResult
    """
    if llm_client is None:
        llm_client = LLMClient()

    default_prompt = (
        "请仔细观察这张屏幕截图，用JSON格式返回：\n"
        "1. description: 用一段话描述屏幕上有什么\n"
        "2. elements: 列出所有可见的按钮、链接、输入框、菜单项等可交互元素，"
        "每个元素包含 text(显示文字)、x(大概水平位置0-100百分比)、y(大概垂直位置0-100百分比)\n"
        "只返回JSON，不要其他文字。"
    )

    for attempt in range(VISION_MAX_RETRIES + 1):
        try:
            result = llm_client.chat_with_vision(
                messages=[{"role": "user", "content": prompt or default_prompt}],
                image_path=image_path,
            )
            raw = result.get("content", "")
            return _parse_vision_response(raw)
        except Exception as e:
            if attempt < VISION_MAX_RETRIES:
                print(f"[Vision] 读屏失败，重试 {attempt+1}/{VISION_MAX_RETRIES}: {e}")
            else:
                print(f"[Vision] 读屏失败: {e}")
                return VisionResult(description=f"视觉读屏失败: {e}", raw_response=str(e))


def vision_find(target: str, image_path: str | None = None,
                llm_client=None) -> tuple[int, int] | None:
    """用 LLM 视觉能力定位屏幕元素（支持图标、图片按钮等非文字元素）

    Args:
        target: 元素描述（如"红色关闭按钮"、"左上角菜单图标"）
        image_path: 截图路径（None 则自动截图）
        llm_client: LLMClient 实例

    Returns:
        (x, y) 像素坐标 或 None
    """
    if llm_client is None:
        llm_client = LLMClient()

    if image_path is None:
        image_path = take_screenshot(tag="vision_find")

    screen_w, screen_h = get_screen_size()

    prompt = (
        f"在屏幕截图中找到 \"{target}\" 的位置。\n"
        "返回JSON格式：{\"x\": 百分比(0-100), \"y\": 百分比(0-100), \"found\": true/false}\n"
        "只返回JSON。"
    )

    try:
        result = llm_client.chat_with_vision(
            messages=[{"role": "user", "content": prompt}],
            image_path=image_path,
        )
        raw = result.get("content", "")

        # 解析 JSON
        json_str = raw
        if "```" in raw:
            json_str = raw.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]

        data = json.loads(json_str.strip())
        if data.get("found", False):
            px = int(data["x"] / 100 * screen_w)
            py = int(data["y"] / 100 * screen_h)
            print(f"[Vision] 找到 \"{target}\" @ ({px}, {py})")
            return (px, py)

        print(f"[Vision] 未找到 \"{target}\"")
        return None
    except Exception as e:
        print(f"[Vision] 视觉定位失败: {e}")
        return None


def _parse_vision_response(raw: str) -> VisionResult:
    """解析 LLM 视觉响应"""
    # 尝试提取 JSON
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
        return VisionResult(
            description=data.get("description", ""),
            elements=data.get("elements", []),
            raw_response=raw,
        )
    except json.JSONDecodeError:
        return VisionResult(description="解析失败", raw_response=raw)


# ============================================================
# 帧差分
# ============================================================

@dataclass
class FrameChange:
    """屏幕变化区域"""
    bbox: tuple[int, int, int, int]   # (left, top, right, bottom)
    center: tuple[int, int]           # 变化区域中心
    changed_pixels: int               # 变化的像素数
    change_ratio: float               # 变化比例 (0~1)

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "center": self.center,
            "changed_pixels": self.changed_pixels,
            "change_ratio": round(self.change_ratio, 4),
        }


def frame_diff(before_path: str, after_path: str,
               threshold: int | None = None,
               min_area: int | None = None) -> list[FrameChange]:
    """对比两帧截图，返回变化区域列表（纯PIL，零新依赖）

    Args:
        before_path: 变化前截图路径
        after_path:  变化后截图路径
        threshold:   像素差异阈值（默认用配置值）
        min_area:    最小变化面积（默认用配置值）

    Returns:
        变化区域列表，按面积从大到小排序
    """
    threshold = threshold or FRAME_DIFF_THRESHOLD
    min_area = min_area or FRAME_DIFF_MIN_AREA

    img_before = Image.open(before_path).convert("RGB")
    img_after = Image.open(after_path).convert("RGB")

    # 确保尺寸一致
    if img_before.size != img_after.size:
        img_after = img_after.resize(img_before.size, Image.LANCZOS)

    w, h = img_before.size
    pixels_before = img_before.load()
    pixels_after = img_after.load()

    # 构建差异热力图
    diff_img = Image.new("L", (w, h), 0)
    diff_pixels = diff_img.load()
    total_changed = 0

    for y in range(h):
        for x in range(w):
            r1, g1, b1 = pixels_before[x, y]
            r2, g2, b2 = pixels_after[x, y]
            diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            if diff > threshold * 3:
                diff_pixels[x, y] = 255
                total_changed += 1

    # 连通区域提取
    regions = _extract_regions(diff_pixels, w, h, min_area)

    changes = []
    for (left, top, right, bottom, count) in regions:
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        changes.append(FrameChange(
            bbox=(left, top, right, bottom),
            center=(center_x, center_y),
            changed_pixels=count,
            change_ratio=count / (w * h),
        ))

    changes.sort(key=lambda c: c.changed_pixels, reverse=True)

    if changes:
        print(f"[Diff] 检测到 {len(changes)} 个变化区域，"
              f"总变化 {total_changed} 像素 ({total_changed/(w*h)*100:.1f}%)")
    else:
        print("[Diff] 屏幕无明显变化")

    return changes


def quick_diff(wait_seconds: float | None = None) -> list[FrameChange]:
    """快速拍两帧做差分（一键调用）

    Args:
        wait_seconds: 两帧之间等待时间（默认用配置值）

    Returns:
        变化区域列表
    """
    wait_seconds = wait_seconds or FRAME_DIFF_WAIT
    before = take_screenshot(save=True, tag="diff_before")
    time.sleep(wait_seconds)
    after = take_screenshot(save=True, tag="diff_after")
    return frame_diff(before, after)


def _extract_regions(pixels, w: int, h: int, min_area: int) -> list[tuple]:
    """从二值差异图提取连通区域（行列扫描法）"""
    row_ranges = []
    for y in range(h):
        left = right = -1
        for x in range(w):
            if pixels[x, y] == 255:
                if left == -1:
                    left = x
                right = x
        if left >= 0:
            row_ranges.append((y, left, right))

    if not row_ranges:
        return []

    # 合并相邻行
    groups = []
    current_group = [row_ranges[0]]

    for i in range(1, len(row_ranges)):
        prev_y, prev_l, prev_r = current_group[-1]
        cur_y, cur_l, cur_r = row_ranges[i]
        if cur_y - prev_y <= 3 and max(cur_l, prev_l) - 5 <= min(cur_r, prev_r):
            current_group.append(row_ranges[i])
        else:
            groups.append(current_group)
            current_group = [row_ranges[i]]
    groups.append(current_group)

    # 每组算 bounding box
    regions = []
    for group in groups:
        ys = [r[0] for r in group]
        ls = [r[1] for r in group]
        rs = [r[2] for r in group]
        top, bottom = min(ys), max(ys)
        left, right = min(ls), max(rs)
        area = (right - left) * (bottom - top)
        if area >= min_area:
            count = sum(r[2] - r[1] + 1 for r in group)
            regions.append((left, top, right, bottom, count))

    return regions
