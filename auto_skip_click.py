#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动检测"跳过"按钮并点击（图像识别版）

原理：持续截取屏幕（或指定区域），用OpenCV模板匹配寻找按钮图片，
匹配到之后用pyautogui模拟真实鼠标点击（系统级点击，不是网页脚本合成事件，
能绕开网站对 event.isTrusted 的校验）。

依赖安装：
    pip install opencv-python pyautogui numpy pillow

使用步骤：
    1. 把要识别的按钮截图保存为 skip_button_template.png，放在本脚本同目录
       （截图要尽量只框住按钮本身，不要带太多背景，这样匹配更准）
    2. 如果按钮位置比较固定，建议设置 SEARCH_REGION，缩小搜索范围，
       既能提速也能减少误识别
    3. 运行: python auto_skip_click.py
    4. 按 Ctrl+C 停止
"""

import time
import sys
import os

import cv2
import numpy as np
import pyautogui

# ========== 配置区 ==========

# 模板图片路径（就是"跳过"按钮的截图）
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skip_button_template.png")

# 匹配置信度阈值，0~1，越接近1要求越严格。
# 如果总是匹配不到，可以调低（比如0.75）；如果经常误触，调高（比如0.9）
MATCH_THRESHOLD = 0.85

# 搜索区域：None表示全屏搜索；如果按钮位置固定，建议设置具体区域以提速+防误触
# 格式: (left, top, width, height)，单位像素，对应你的屏幕/浏览器窗口坐标
# 例如按钮总是在浏览器窗口右下角，可以设一个大致框住那块区域的矩形
SEARCH_REGION = None
# 示例（需要根据你自己屏幕分辨率和浏览器窗口位置调整）：
# SEARCH_REGION = (1000, 600, 900, 350)

# 扫描间隔（秒）
SCAN_INTERVAL = 0.5

# 点击后冷却时间（秒），避免短时间重复点击同一个位置
CLICK_COOLDOWN = 2.0

# 是否在点击前先移动鼠标过去再点击（更像真实操作，也更保险）
MOVE_BEFORE_CLICK = True

# 点击后鼠标移动的持续时间（秒），0表示瞬移
MOVE_DURATION = 0.15

# ========== 配置区结束 ==========


def load_template():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"[错误] 找不到模板图片: {TEMPLATE_PATH}")
        print("请把按钮截图保存为 skip_button_template.png 放在脚本同目录下")
        sys.exit(1)

    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
    if template is None:
        print(f"[错误] 无法读取模板图片: {TEMPLATE_PATH}")
        sys.exit(1)

    h, w = template.shape[:2]
    print(f"[信息] 模板图片加载成功，尺寸: {w}x{h}")
    return template


def screenshot_to_cv(region=None):
    """截图并转换为OpenCV格式(BGR)"""
    shot = pyautogui.screenshot(region=region)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    return frame


def find_button(frame, template, threshold):
    """在frame中用模板匹配寻找template，返回最佳匹配的中心坐标和置信度，找不到返回None"""
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None, max_val

    h, w = template.shape[:2]
    center_x = max_loc[0] + w // 2
    center_y = max_loc[1] + h // 2
    return (center_x, center_y), max_val


def click_at(x, y, region_offset=(0, 0)):
    """点击屏幕坐标(x, y)，如果截图时用了region，需要加上偏移换算成真实屏幕坐标"""
    real_x = x + region_offset[0]
    real_y = y + region_offset[1]

    if MOVE_BEFORE_CLICK:
        pyautogui.moveTo(real_x, real_y, duration=MOVE_DURATION)
    pyautogui.click(real_x, real_y)
    print(f"[点击] 坐标 ({real_x}, {real_y})")


def main():
    print("=" * 50)
    print("自动跳过按钮点击器（图像识别版）启动")
    print(f"匹配阈值: {MATCH_THRESHOLD}")
    print(f"搜索区域: {'全屏' if SEARCH_REGION is None else SEARCH_REGION}")
    print("按 Ctrl+C 停止")
    print("=" * 50)

    template = load_template()
    region_offset = (SEARCH_REGION[0], SEARCH_REGION[1]) if SEARCH_REGION else (0, 0)

    last_click_time = 0

    try:
        while True:
            frame = screenshot_to_cv(region=SEARCH_REGION)
            match_pos, confidence = find_button(frame, template, MATCH_THRESHOLD)

            if match_pos is not None:
                now = time.time()
                if now - last_click_time >= CLICK_COOLDOWN:
                    print(f"[发现] 匹配到按钮，置信度={confidence:.3f}，位置={match_pos}")
                    click_at(match_pos[0], match_pos[1], region_offset)
                    last_click_time = now
                else:
                    print(f"[跳过点击] 冷却中，置信度={confidence:.3f}")
            else:
                # 置信度太低时不打印，避免刷屏；调试时可以取消下面这行注释
                # print(f"[未匹配] 当前最高置信度={confidence:.3f}")
                pass

            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n[停止] 用户中断，程序退出")


if __name__ == "__main__":
    main()
