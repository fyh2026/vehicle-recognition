# Vehicle Detection + Classification + Counting System

A traffic video vehicle recognition system based on **YOLO11 + ByteTrack + OpenCV**.

## Tech Stack

| Layer | Technology | Description |
|------|------------|-------------|
| Detection & Classification | YOLO11 (Ultralytics) | COCO pretrained model, single inference outputs both location and vehicle type |
| Multi-Object Tracking | ByteTrack (supervision) | Motion-based tracking, assigns a unique ID to each vehicle |
| Counting | Virtual counting line + cross-product direction | Geometric algorithm determines crossing direction (IN / OUT) |
| Video Processing | OpenCV | Codec, window display, annotation rendering |

## Vehicle Types

| Class | English | COCO ID |
|------|------|---------|
| 轿车 | car | 2 |
| 摩托车 | motorcycle | 3 |
| 公交车 | bus | 5 |
| 卡车 | truck | 7 |

## Dependencies

- Python 3.9+
- PyTorch (CPU version works)

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

## Quick Start

```bash
# Basic run (auto-downloads yolo11n.pt on first run)
python vehicle_detect.py --source your_video.mp4

# Use a larger model (more accurate but slower)
python vehicle_detect.py --source test.mp4 --model yolo11m.pt

# Debug mode (show detection details)
python vehicle_detect.py --source test.mp4 --debug
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | Required | Input video path |
| `--model` | `yolo11n.pt` | YOLO model (n→s→m→l→x, bigger = more accurate & slower) |
| `--conf` | `0.3` | Detection confidence threshold (0~1) |
| `--skip` | `2` | Detect every N frames (1 = frame-by-frame) |
| `--resize` | `640` | Resize width before detection (0 = no resize) |
| `--line` | `20,50,80,50` | Counting line coordinates x1,y1,x2,y2 (percentage) |
| `--debug` | Off | Print all detected classes every 30 frames |

## Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| Space | Pause / Resume |

On exit, automatically outputs `<video_name>_log.csv` — recording frame number, vehicle ID, type, and direction for each crossing event.

## Project Structure

```
vehicle-recognition/
├── vehicle_detect.py    # Main program (~280 lines)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── comparison.md       # Technical comparison with face recognition project (English)
└── 对比.md             # Technical comparison with face recognition project (Chinese)
```

## How It Works

```
Video frame → Resize → YOLO detection → Filter vehicle classes → ByteTrack tracking
                                                                   │
                              ┌─────────────────────────────────────┘
                              ▼
                    Cross-product direction detection
                              │
                         ┌────┴────┐
                         ▼         ▼
                      IN count   OUT count
                         │         │
                         └────┬────┘
                              ▼
                   OpenCV annotation + live display
                              │
                              ▼
                        CSV event log
```

- **Frame-skip acceleration**: Detect once every N frames, reuse results for skipped frames
- **Resize acceleration**: Downscale before detection, significantly reduces inference time
- **Cross-product direction**: Compute the cross-product sign of the detection box bottom-center relative to the counting line; a sign flip indicates a crossing event

## Credits

Standing on the shoulders of giants. Thanks to the following open-source projects:

| Project | Author/Org | License | Purpose |
|---------|-----------|---------|---------|
| [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11) | Ultralytics | AGPL-3.0 | Object detection & classification model |
| [supervision](https://github.com/roboflow/supervision) | Roboflow | MIT | ByteTrack tracking, annotation tools, LineZone counting |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | ifzhang | MIT | Multi-object tracking algorithm |
| [OpenCV](https://github.com/opencv/opencv-python) | OpenCV team | Apache 2.0 | Video processing & rendering |
| [NumPy](https://github.com/numpy/numpy) | NumPy community | BSD-3-Clause | Array operations & data processing |
| [PyTorch](https://github.com/pytorch/pytorch) | Meta / Linux Foundation | BSD | Deep learning inference engine |

**Respect open source, give back to open source** — If you find this project helpful, consider giving a ⭐ to the projects above too.
