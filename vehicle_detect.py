"""
车辆检测 + 车型分类 + 计数系统
技术栈: YOLO11 + ByteTrack + supervision + OpenCV
用于毕业答辩 / 课程项目展示
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ============================================================
#  配置常量
# ============================================================

# COCO 数据集中车辆相关的类别 ID → (英文名, 中文标签)
VEHICLE_CLASSES: dict[int, tuple[str, str]] = {
    2: ("car",        "轿车"),
    3: ("motorcycle", "摩托车"),
    5: ("bus",        "公交车"),
    7: ("truck",      "卡车"),
}

# 不同车型的标注颜色 (BGR)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    2: (0, 255, 0),      # 轿车 绿色
    3: (255, 0, 255),    # 摩托车 品红
    5: (0, 255, 255),    # 公交车 黄色
    7: (255, 0, 0),      # 卡车 蓝色
}


# ============================================================
#  VehicleCounter — 主流水线
# ============================================================

class VehicleCounter:
    """车辆检测 + 跟踪 + 计数 流水线"""

    def __init__(self, source: str, model_name: str, conf: float,
                 line_coords: tuple[float, float, float, float],
                 skip: int = 1, resize: int = 0, debug: bool = False) -> None:
        self.source = source
        self.conf = conf
        self.skip = skip
        self.resize = resize
        self.debug = debug
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {source}")

        # YOLO 模型 (首次运行自动下载 .pt 权重文件)
        print(f"[INFO] 加载模型: {model_name} ...")
        self.model = YOLO(model_name)
        print(f"[INFO] 模型就绪 (skip={skip}, resize={resize or '无'})")

        # ByteTrack 跟踪器
        self.tracker = sv.ByteTrack()

        # ---- 计数线 (百分比坐标 → 像素坐标) ----
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.line_start = sv.Point(
            int(line_coords[0] * w / 100),
            int(line_coords[1] * h / 100),
        )
        self.line_end = sv.Point(
            int(line_coords[2] * w / 100),
            int(line_coords[3] * h / 100),
        )
        self.line_zone = sv.LineZone(start=self.line_start, end=self.line_end)

        # 标注器
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=2, text_thickness=2, text_scale=1,
        )
        self.box_annotator = sv.BoxAnnotator(thickness=2)

        # ---- 按车型计数 ----
        self.in_by_class: dict[int, int] = defaultdict(int)
        self.out_by_class: dict[int, int] = defaultdict(int)

        # 记录每个 track_id 上一帧在线哪一侧, 用于判定穿越
        self._prev_side: dict[int, int] = {}   # track_id → side (-1/0/1)

        # 事件日志
        self.events: list[tuple[int, int, str, str]] = []

    # --------------------------------------------------------

    def _filter_vehicles(self, results, frame_idx: int = 0) -> sv.Detections:
        """从 YOLO 结果中提取车辆检测"""
        detections = sv.Detections.from_ultralytics(results[0])
        if self.debug and len(detections) > 0 and frame_idx % 30 == 0:
            # COCO 80 类名映射
            coco_names = results[0].names
            print(f"\n[DEBUG 帧{frame_idx}] 检测到 {len(detections)} 个目标:")
            for i in range(min(len(detections), 10)):
                cid = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                name = coco_names.get(cid, f"class_{cid}")
                marker = "← 车辆" if cid in VEHICLE_CLASSES else ""
                print(f"  class_id={cid:2d}  {name:<15s}  conf={conf:.2f}  {marker}")
        if len(detections) == 0:
            return detections
        mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
        return detections[mask]

    def _detect_crossings(self, detections: sv.Detections, frame_idx: int) -> None:
        """检测每辆车是否跨线, 更新 per-class 计数和事件日志"""
        if detections.tracker_id is None:
            return

        lx, ly = self.line_start.x, self.line_start.y
        ex, ey = self.line_end.x, self.line_end.y

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            class_id = int(detections.class_id[i])
            # 取检测框底部中心作为参考点
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = (x1 + x2) / 2
            cy = y2   # 底边

            # 叉积判定方向
            cross = (ex - lx) * (cy - ly) - (ey - ly) * (cx - lx)
            side = 1 if cross > 5 else (-1 if cross < -5 else 0)

            prev = self._prev_side.get(tid)
            if prev is not None and prev != 0 and side != 0 and prev != side:
                # 发生了穿越
                name_en = VEHICLE_CLASSES.get(class_id, ("unknown", "未知"))[0]
                direction = "IN" if (prev < 0 and side > 0) else "OUT"
                if direction == "IN":
                    self.in_by_class[class_id] += 1
                else:
                    self.out_by_class[class_id] += 1
                self.events.append((frame_idx, tid, name_en, direction))

            if side != 0:
                self._prev_side[tid] = side

    # --------------------------------------------------------

    def run(self) -> None:
        """主循环"""
        frame_idx = 0
        last_detections: sv.Detections | None = None
        print(f"[INFO] 开始处理: {self.source}")
        print("[INFO] 按 Q 键退出, 按空格暂停")

        paused = False

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # ---- 缩放 ----
                display_frame = frame
                if self.resize > 0:
                    scale = self.resize / frame.shape[1]
                    detect_frame = cv2.resize(frame, (self.resize, int(frame.shape[0] * scale)))
                else:
                    detect_frame = frame

                # ---- 检测（跳帧） ----
                if frame_idx % self.skip == 0:
                    # 1. YOLO11 检测
                    results = self.model(detect_frame, conf=self.conf, iou=0.5, verbose=False)
                    detections = self._filter_vehicles(results, frame_idx)

                    # 坐标映射回原图尺寸
                    if self.resize > 0 and len(detections) > 0:
                        inv_scale = frame.shape[1] / self.resize
                        detections.xyxy[:, [0, 2]] *= inv_scale
                        detections.xyxy[:, [1, 3]] *= inv_scale

                    # 2. ByteTrack 跟踪
                    detections = self.tracker.update_with_detections(detections)

                    # 3. 穿越检测 + per-class 计数
                    self._detect_crossings(detections, frame_idx)

                    # 4. LineZone 总计数
                    self.line_zone.trigger(detections)

                    last_detections = detections
                else:
                    detections = last_detections

                # 5. 标注检测框 + 标签
                if detections is not None and detections.tracker_id is not None and len(detections) > 0:
                    frame = self.box_annotator.annotate(
                        scene=frame, detections=detections,
                    )
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        tid = int(detections.tracker_id[i])
                        cid = int(detections.class_id[i])
                        name_cn = VEHICLE_CLASSES.get(cid, ("?", "?"))[0]  # 英文名 (OpenCV 不支持中文字体)
                        label = f"#{tid} {name_cn}"
                        color = CLASS_COLORS.get(cid, (255, 255, 255))
                        cv2.putText(frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 6. 画计数线 + 统计面板
                self.line_annotator.annotate(frame=frame, line_counter=self.line_zone)
                self._draw_panel(frame)

                frame_idx += 1

            # 显示
            cv2.imshow("Vehicle Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

        # 收尾
        self.cap.release()
        cv2.destroyAllWindows()
        self._save_csv()
        self._print_summary(frame_idx)

    # --------------------------------------------------------
    #  画面绘制
    # --------------------------------------------------------

    def _draw_panel(self, frame: np.ndarray) -> None:
        """在画面右上角绘制半透明统计面板"""
        pw, ph = 270, 240
        px = frame.shape[1] - pw - 10
        py = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y = py + 28
        cv2.putText(frame, "Vehicle Counter", (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y += 35
        cv2.putText(frame, f"IN : {self.line_zone.in_count}", (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        y += 28
        cv2.putText(frame, f"OUT: {self.line_zone.out_count}", (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        y += 38
        cv2.putText(frame, "By Class", (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for cid, (name_en, _) in VEHICLE_CLASSES.items():
            y += 22
            color = CLASS_COLORS.get(cid, (255, 255, 255))
            cv2.putText(frame,
                        f"  {name_en}: {self.in_by_class[cid]}/{self.out_by_class[cid]}",
                        (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # --------------------------------------------------------
    #  输出
    # --------------------------------------------------------

    def _save_csv(self) -> None:
        """保存穿越事件 CSV"""
        if not self.events:
            return
        csv_path = Path(self.source).stem + "_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "track_id", "class", "direction"])
            writer.writerows(self.events)
        print(f"[CSV] 事件日志已保存: {csv_path} ({len(self.events)} 条穿越记录)")

    def _print_summary(self, total_frames: int) -> None:
        """控制台汇总"""
        print(f"\n{'='*40}")
        print(f"  处理完成 — 共 {total_frames} 帧")
        print(f"  总计  IN : {self.line_zone.in_count}")
        print(f"       OUT : {self.line_zone.out_count}")
        print(f"  车型分布:")
        for cid, (name_en, _) in VEHICLE_CLASSES.items():
            print(f"    {name_en:<10s} IN {self.in_by_class[cid]:>4d}  OUT {self.out_by_class[cid]:>4d}")
        print(f"{'='*40}")


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="车辆检测 + 车型分类 + 计数 (YOLO11 + ByteTrack)",
    )
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO 模型 (nano 最快→m→l→x 最准)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="检测置信度阈值 (默认 0.3)")
    parser.add_argument("--line", default="20,50,80,50",
                        help="计数线 x1,y1,x2,y2 百分比坐标")
    parser.add_argument("--skip", type=int, default=2,
                        help="每 N 帧检测一次 (默认 2，1=逐帧)")
    parser.add_argument("--resize", type=int, default=640,
                        help="检测前缩放宽度 (默认 640，0=不缩放)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式: 打印所有检测到的类别")
    args = parser.parse_args()

    coords = tuple(map(float, args.line.split(",")))
    vc = VehicleCounter(args.source, args.model, args.conf, coords,
                        skip=args.skip, resize=args.resize, debug=args.debug)
    vc.run()
