# 车辆检测 + 车型分类 + 计数系统

基于 **YOLO11 + ByteTrack + OpenCV** 的交通视频车辆识别系统。

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 检测 & 分类 | YOLO11 (Ultralytics) | COCO 预训练模型，一次推理同时出位置 + 车型 |
| 多目标跟踪 | ByteTrack (supervision) | 纯运动模型跟踪，为每辆车分配唯一 ID |
| 计数 | 虚拟计数线 + 叉积方向判定 | 几何算法判定车辆穿越方向（IN / OUT） |
| 视频处理 | OpenCV | 编解码、窗口显示、标注绘制 |

## 识别的车型

| 类别 | 英文 | COCO ID |
|------|------|---------|
| 轿车 | car | 2 |
| 摩托车 | motorcycle | 3 |
| 公交车 | bus | 5 |
| 卡车 | truck | 7 |

## 环境依赖

- Python 3.9+
- PyTorch (CPU 版即可)

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
ultralytics>=8.3.0
supervision>=0.23.0
opencv-python>=4.9.0
numpy>=1.26.0
torch>=2.1.0
```

## 快速开始

```bash
# 基本运行（首次自动下载 yolo11n.pt）
python vehicle_detect.py --source 你的视频.mp4

# 使用更大模型（更准但更慢）
python vehicle_detect.py --source test.mp4 --model yolo11m.pt

# 调试模式（查看检测详情）
python vehicle_detect.py --source test.mp4 --debug
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必填 | 输入视频路径 |
| `--model` | `yolo11n.pt` | YOLO 模型（n→s→m→l→x，越大越准越慢） |
| `--conf` | `0.3` | 检测置信度阈值（0~1） |
| `--skip` | `2` | 每 N 帧检测一次（1=逐帧检测） |
| `--resize` | `640` | 检测前缩放宽度（0=不缩放） |
| `--line` | `20,50,80,50` | 计数线坐标 x1,y1,x2,y2（百分比） |
| `--debug` | 关闭 | 每 30 帧打印检测到的所有类别 |

## 操作说明

| 按键 | 功能 |
|------|------|
| Q | 退出程序 |
| 空格 | 暂停 / 继续 |

退出后自动输出 `视频名_log.csv`——记录每次穿越事件的帧号、车辆ID、车型、方向。

## 项目结构

```
车辆识别/
├── vehicle_detect.py    # 主程序（~280 行）
├── requirements.txt     # Python 依赖
├── README.md           # 本文件
└── 对比.md             # 与人脸识别项目的技术对比
```

## 工作原理

```
视频帧 → 缩放 → YOLO 检测 → 过滤车辆类别 → ByteTrack 跟踪
                                              │
                         ┌────────────────────┘
                         ▼
              叉积判定跨线方向
                         │
                    ┌────┴────┐
                    ▼         ▼
                 IN 计数   OUT 计数
                    │         │
                    └────┬────┘
                         ▼
              OpenCV 标注 + 实时显示
                         │
                         ▼
                   CSV 事件日志
```

- **跳帧加速**：每 N 帧检测一次，其余帧复用上次结果
- **缩放加速**：检测前缩小分辨率，大幅降低推理时间
- **叉积方向判定**：计算检测框底边中点相对于计数线的叉积符号，符号翻转即判定穿越

## 开源致谢

本项目站在巨人的肩膀上，感谢以下开源项目的贡献：

| 项目 | 作者/组织 | 许可证 | 用途 |
|------|----------|--------|------|
| [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11) | Ultralytics | AGPL-3.0 | 目标检测与分类模型 |
| [supervision](https://github.com/roboflow/supervision) | Roboflow | MIT | ByteTrack 跟踪、标注工具、LineZone 计数 |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | ifzhang | MIT | 多目标跟踪算法 |
| [OpenCV](https://github.com/opencv/opencv-python) | OpenCV team | Apache 2.0 | 视频处理与画面绘制 |
| [NumPy](https://github.com/numpy/numpy) | NumPy community | BSD-3-Clause | 数组运算与数据处理 |
| [PyTorch](https://github.com/pytorch/pytorch) | Meta / Linux Foundation | BSD | 深度学习推理引擎 |

**尊重开源，回馈开源** —— 如果你觉得这个项目有帮助，也请给上面的开源项目点个 ⭐
