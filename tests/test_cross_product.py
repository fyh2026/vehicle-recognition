"""叉积跨线判定算法的单元测试"""

import sys
from pathlib import Path

# 将项目根目录加入路径以便导入 vehicle_detect
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import supervision as sv

from vehicle_detect import VehicleCounter


# ------------------------------------------------------------
#  模拟一个 VehicleCounter 实例用于测试（接枝构造函数）
# ------------------------------------------------------------

@pytest.fixture
def counter(mocker):
    """创建一个轻量 VehicleCounter，跳过视频加载和模型初始化"""
    # 用 monkeypatch 模式手动构造最小实例
    # 这里直接测核心算法逻辑，不依赖 OpenCV/视频
    ...


# ------------------------------------------------------------
#  纯函数测试：叉积方向判定
# ------------------------------------------------------------

def test_cross_product_side():
    """验证叉积符号判断点在线左/线右"""
    from vehicle_detect import VEHICLE_CLASSES

    # 手动模拟 _detect_crossings 中的叉积逻辑
    lx, ly = 0, 50
    ex, ey = 100, 50       # 水平线 y=50, 从左到右
    eps = 0.1

    def _side(cx, cy):
        cross = (ex - lx) * (cy - ly) - (ey - ly) * (cx - lx)
        return 1 if cross > eps else (-1 if cross < -eps else 0)

    # 点在线上方（北边）→ 叉积为负
    assert _side(50, 40) == -1, "点在线之上应为 -1"

    # 点在线下方（南边）→ 叉积为正
    assert _side(50, 60) == 1, "点在线之下应为 1"

    # 点正好在线上 → 叉积为 0
    assert _side(50, 50) == 0, "点正好在线应为 0"

    # 垂直线场景
    def _side_v(cx, cy):
        lx, ly = 50, 0
        ex, ey = 50, 100   # 垂直线 x=50
        cross = (ex - lx) * (cy - ly) - (ey - ly) * (cx - lx)
        return 1 if cross > eps else (-1 if cross < -eps else 0)

    # 点在垂线右侧
    assert _side_v(60, 50) == 1, "垂线右侧应为 1"

    # 点在垂线左侧
    assert _side_v(40, 50) == -1, "垂线左侧应为 -1"


def test_crossing_detection():
    """模拟两次检测，验证穿越触发的方向判断"""
    # 水平线 y=50
    eps = 0.1
    lx, ly = 0, 50
    ex, ey = 100, 50

    prev_side = {}  # 模拟 _prev_side

    def update(tid: int, cx: float, cy: float) -> str | None:
        cross = (ex - lx) * (cy - ly) - (ey - ly) * (cx - lx)
        side = 1 if cross > eps else (-1 if cross < -eps else 0)

        direction = None
        prev = prev_side.get(tid)
        if prev is not None and prev != 0 and side != 0 and prev != side:
            direction = "IN" if (prev < 0 and side > 0) else "OUT"

        if side != 0:
            prev_side[tid] = side
        return direction

    # 车辆第 1 帧：在线之上（北边）
    assert update(1, 50, 40) is None, "第一帧无穿越"

    # 车辆第 2 帧：跨到线之下（南边）→ IN（从上往下）
    assert update(1, 50, 60) == "IN", "从上往下应为 IN"

    # 车辆第 2 帧：又回到北边 → OUT（从下往上）
    assert update(1, 50, 30) == "OUT", "从下往上应为 OUT"


def test_cross_eps_threshold():
    """验证叉积容差阈值能过滤抖动"""
    eps = 50  # 用大阈值模拟图像归一化后的效果
    lx, ly = 0, 50
    ex, ey = 100, 50

    def _side(cx, cy):
        cross = (ex - lx) * (cy - ly) - (ey - ly) * (cx - lx)
        return 1 if cross > eps else (-1 if cross < -eps else 0)

    # 接近线但未超过阈值
    assert _side(50, 50.4) == 0, "接近线的抖动应被忽略"
    assert _side(50, 49.6) == 0, "接近线的抖动应被忽略"
