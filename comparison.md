# Project Comparison: Face Recognition vs Vehicle Recognition

## Overview

| | Face Recognition | Vehicle Recognition |
|---|---|---|
| Code Size | 5 files / ~600 lines | 1 file / ~280 lines |
| Dependencies | 2 (opencv, numpy) | 5 (YOLO + supervision + torch + opencv + numpy) |

## Technology Gap

```
Face Recognition:  Haar Cascade + LBPH  →  2010s Traditional CV
Vehicle Recognition: YOLO11 + ByteTrack  →  2024-25 Deep Learning
                    ↑ Two generations apart
```

## Detailed Comparison

| Dimension | Face Recognition (Old) | Vehicle Recognition (New) |
|-----------|:---------------------:|:------------------------:|
| Approach | Haar Cascade + LBPH (Local Binary Patterns) | YOLO11 one-stage detection + ByteTrack multi-object tracking |
| Learning Curve | Collect 100-200 faces → Train model → Then use | Pretrained model, zero training, plug-and-play |
| Accuracy | LBPH heavily affected by lighting/angle, low accuracy ceiling | Deep learning model, COCO pretrained, far superior accuracy |
| GPU Acceleration | ❌ CPU only | ✅ torch supports GPU (CPU also works) |
| Compatibility | ❌ OpenCV 4.9+ removed `cv2.face`, requires `opencv-contrib` | ✅ All mainstream libraries, no deprecation issues |
| Recognition | Face detection + identity matching only | Detection + vehicle classification (car/motorcycle/bus/truck) |
| Counting | None | Virtual line bidirectional counting (IN/OUT) + per-class stats |
| Data Output | None | CSV event log |
| Real-time Experience | Decent framerate but slow recognition | Frame-skip + resize optimization, smooth playback |
| Demo Effect | Static recognition, terminal menu interaction | Real-time annotated video + counting panel, visually impressive |

## Summary

The vehicle recognition project improves upon the face recognition project by:
- **Modern technology** — upgraded from 2010s traditional CV to 2024 deep learning
- **Easier to use** — no data collection or model training needed, runs out of the box
- **Higher accuracy** — YOLO deep learning vs LBPH traditional features
- **Better presentation** — real-time video vs terminal menus
