"""Rule-compliant RGB detection UI for the national competition.

The UI deliberately exposes no runtime parameter controls. It watches the
existing Astra RGB JPEG and worker CSV, draws class-colored boxes, and shows
ID/name/count/confidence plus status, elapsed time, and table number.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

from national_competition.config import GOAL_IDS, NAMES, TABLE_CLASS_ID


COLORS = (
    "#36C5F0", "#F59E0B", "#A78BFA", "#FB7185",
    "#10B981", "#F97316", "#84CC16", "#EAB308",
    "#06B6D4", "#8B5CF6", "#3B82F6", "#EC4899",
    "#EF4444", "#F97316", "#22C55E", "#D4A72C", "#FFFFFF",
)


@dataclass(frozen=True)
class UiDetection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    depth_state: str = "UNKNOWN"
    capacity_scores: tuple[float, float] | None = None


def _font(size: int):
    candidates = (
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_18 = _font(18)
FONT_22 = _font(22)
FONT_28 = _font(28)
FONT_36 = _font(36)


def read_worker_csv(path: Path):
    """Read sequence then cls,conf,x1,y1,x2,y2[,score350,score550]."""
    if not path.exists() or path.stat().st_size == 0:
        return None, []
    return parse_worker_csv(path.read_text(encoding="ascii", errors="replace"))


def parse_worker_csv(text):
    lines = text.splitlines()
    if not lines:
        return None, []
    try:
        sequence = int(lines[0].strip())
    except ValueError:
        sequence = None
    detections = []
    for row in csv.reader(lines[1:]):
        if len(row) < 6:
            continue
        try:
            class_id = int(row[0])
            confidence = float(row[1])
            box = tuple(float(value) for value in row[2:6])
        except ValueError:
            continue
        if 0 <= class_id < len(NAMES):
            capacity_scores = None
            if class_id in (10, 11) and len(row) >= 8:
                try:
                    candidate = tuple(float(value) for value in row[6:8])
                    if all(0. <= value <= 1. for value in candidate):
                        capacity_scores = candidate
                except ValueError:
                    pass
            detections.append(UiDetection(class_id, confidence, box, capacity_scores=capacity_scores))
    return sequence, detections


def aggregate(detections):
    grouped = defaultdict(list)
    for detection in detections:
        if detection.class_id != TABLE_CLASS_ID:
            grouped[detection.class_id].append(detection.confidence)
    return [(class_id, len(scores), sorted(scores, reverse=True))
            for class_id, scores in sorted(grouped.items())]


def confidence_text(count, confidences):
    return f"{count} / " + "/".join(f"{confidence:.2f}" for confidence in confidences)


def depth_to_image(depth):
    import cv2
    import numpy as np
    valid=np.isfinite(depth)&(depth>250)&(depth<5000)
    value=np.nan_to_num(depth,nan=0.)
    normalized=(np.clip((value-250.)/2750.,0.,1.)*255).astype(np.uint8)
    colored=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
    colored[~valid]=0
    return Image.fromarray(cv2.cvtColor(colored,cv2.COLOR_BGR2RGB))


def render_dashboard(frame: Image.Image, detections, *, table: int, status: str,
                     elapsed: float, model_version: str, output_path: str = "",
                     summary_detections=None, detail: str = "", ocr_summary: str = "", extra_items=None,
                     depth_frame=None, depth_detail='',rgb_preview=None,preview_to_reference=None,
                     preview_crop=None,round_table_outline=False,table_label=None,square_table_polygon=None):
    frame = frame.convert("RGB")
    target_h = 720
    dual=depth_frame is not None
    enlarged_round=dual and round_table_outline
    rgb_width,rgb_height=(600,450) if enlarged_round else (490,368)
    depth_x=630 if enlarged_round else 520
    depth_width=390 if enlarged_round else 490
    crop_x,crop_y=0,0
    if dual and preview_crop is not None:
        crop_x,crop_y,crop_right,crop_bottom=preview_crop
        frame=frame.crop(preview_crop)
        depth_frame=depth_frame.crop(preview_crop)
    elif dual and any(d.class_id==TABLE_CLASS_ID for d in detections):
        boxes=[d.xyxy for d in detections]
        crop_x=max(0,int(min(b[0] for b in boxes))-16)
        crop_y=max(0,int(min(b[1] for b in boxes))-20)
        crop_right=min(frame.width,int(max(b[2] for b in boxes))+16)
        crop_bottom=min(frame.height,int(max(b[3] for b in boxes))+20)
        crop=(crop_x,crop_y,crop_right,crop_bottom)
        frame=frame.crop(crop)
        depth_frame=depth_frame.crop(crop)
    scale = min((rgb_width if dual else 1000) / frame.width,(rgb_height if dual else target_h)/frame.height)
    depth_scale=min(depth_width/frame.width,rgb_height/frame.height) if dual else scale
    display = frame.resize((round(frame.width * scale), round(frame.height * scale)), Image.Resampling.BILINEAR)
    high_preview=rgb_preview is not None and preview_to_reference is not None
    if high_preview:
        import cv2
        import numpy as np
        reference_to_panel=np.array([[scale,0,-crop_x*scale],[0,scale,-crop_y*scale],[0,0,1]],dtype=float)
        transformed=cv2.warpPerspective(rgb_preview,reference_to_panel@preview_to_reference,
                                        display.size,flags=cv2.INTER_CUBIC)
        display=Image.fromarray(cv2.cvtColor(transformed,cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", (1360, 800), "#061523")
    draw = ImageDraw.Draw(canvas)
    if ocr_summary:
        draw.text((28,24),ocr_summary,font=FONT_18,fill='#FFD166')
    ox, oy = (18,90) if dual else (28,58)
    canvas.paste(display, (ox, oy))
    if dual:
        draw.text((18,60),'RGB（高清预览）' if high_preview else 'RGB',font=FONT_18,fill='#EAF8FF')
        draw.text((depth_x,60),'深度（同一配对帧）',font=FONT_18,fill='#EAF8FF')
        depth_size=(round(frame.width*depth_scale),round(frame.height*depth_scale))
        depth_display=depth_frame.convert('RGB').resize(depth_size,Image.Resampling.NEAREST)
        canvas.paste(depth_display,(depth_x,oy))
        draw.text((18,565 if enlarged_round else 485),'绿：三维结构    红：可靠平面    青：透明支持    黄：不确定',font=FONT_18,fill='#EAF8FF')
        draw.text((18,605 if enlarged_round else 525),depth_detail,font=FONT_18,fill='#FFD166')

    for detection in detections:
        bx1,by1,bx2,by2=detection.xyxy
        x1,y1,x2,y2=(bx1-crop_x)*scale,(by1-crop_y)*scale,(bx2-crop_x)*scale,(by2-crop_y)*scale
        color = ('#FFFFFF' if detection.class_id==16 else
                 {'PASS':'#38FF70','FLAT':'#FF3030','TRANSPARENT':'#36C5F0','UNKNOWN':'#FFD166'}.get(detection.depth_state,'#FFD166')) if dual else COLORS[detection.class_id]
        box = (ox + x1, oy + y1, ox + x2, oy + y2)
        table_ellipse=round_table_outline and detection.class_id==TABLE_CLASS_ID
        outline=draw.ellipse if table_ellipse else draw.rectangle
        table_polygon=square_table_polygon if detection.class_id==TABLE_CLASS_ID else None
        if table_polygon:
            points=[(ox+(x-crop_x)*scale,oy+(y-crop_y)*scale) for x,y in table_polygon]
            draw.line(points+[points[0]],fill=color,width=4,joint='curve')
        else:outline(box, outline=color, width=4)
        if dual:
            depth_box=(depth_x+(bx1-crop_x)*depth_scale,oy+(by1-crop_y)*depth_scale,
                       depth_x+(bx2-crop_x)*depth_scale,oy+(by2-crop_y)*depth_scale)
            if table_polygon:
                points=[(depth_x+(x-crop_x)*depth_scale,oy+(y-crop_y)*depth_scale) for x,y in table_polygon]
                draw.line(points+[points[0]],fill='#07111C',width=4,joint='curve')
                draw.line(points+[points[0]],fill=color,width=2,joint='curve')
            else:
                outline(depth_box,outline='#07111C',width=4)
                outline(depth_box,outline=color,width=2)
        goal = GOAL_IDS[detection.class_id] or "TABLE"
        label = NAMES[detection.class_id].replace('瓶装饮用水','水') if dual else NAMES[detection.class_id]
        label_font=_font(20 if enlarged_round else 16) if dual else FONT_18
        text_box = draw.textbbox((0, 0), label, font=label_font)
        lh = text_box[3] - text_box[1] + 8
        lx, ly = int(box[0]), max(oy, int(box[1]) - lh)
        draw.text((lx + 6, ly + 3), label, font=label_font, fill=color,
                  stroke_width=1, stroke_fill="#04111C")

    panel_x = 1050
    draw.rounded_rectangle((1025, 28, 1334, 772), radius=18, fill="#0B263D", outline="#1FC7FF", width=2)
    draw.text((panel_x, 55), table_label or "3D视觉识别", font=FONT_28, fill="#EAF8FF")
    status_color = "#3DFF9A" if status == "识别完成" else "#FFD166"
    draw.text((panel_x, 105), f"状态：{status}", font=FONT_22, fill=status_color)
    draw.text((panel_x, 140), f"已用时间：{elapsed:.1f}s", font=FONT_18, fill="#D7ECF7")
    draw.text((panel_x, 170), f"当前桌位：{table_label}" if table_label else f"当前桌号：Table {table}", font=FONT_18, fill="#D7ECF7")
    draw.line((1045, 205, 1315, 205), fill="#2C6685", width=2)
    draw.text((1048, 220), "ID", font=FONT_18, fill="#8EDCFF")
    draw.text((1120, 220), "名称", font=FONT_18, fill="#8EDCFF")
    draw.text((1222, 220), "数量/置信度", font=_font(14), fill="#8EDCFF")
    y = 255
    panel_rows = detections if summary_detections is None else summary_detections
    display_rows = [(GOAL_IDS[cls],NAMES[cls],count,confidences,COLORS[cls])
                    for cls,count,confidences in aggregate(panel_rows)]
    display_rows += [(item.get('goal_id') or '待公布',item['name'],item['count'],
                      [item.get('confidence',0.)],'#EAF8FF') for item in extra_items or []]
    row_height=min(24.,370./max(1,len(display_rows)))
    row_font=_font(14 if row_height>=18 else 11)
    for goal, name, count, confidences, color in display_rows:
        draw.text((1048, y), goal, font=row_font, fill=color)
        if len(name) > 7:
            name = name[:7] + "…"
        draw.text((1120, y), name, font=row_font, fill="#F4FAFD")
        confidence_label=confidence_text(count,confidences)
        confidence_font=row_font
        for size in (12,11,10,9,8):
            candidate=_font(size)
            if draw.textbbox((0,0),confidence_label,font=candidate)[2]<=125:
                confidence_font=candidate;break
        draw.text((1202, y), confidence_label, font=confidence_font, fill="#F4FAFD")
        y += row_height
    if detail:
        draw.text((1048, 645), detail, font=_font(14), fill="#FFD166")
    draw.text((1050, 684), f"模型：{model_version}", font=_font(12), fill="#9AB9C9")
    if output_path:
        draw.text((1050, 714), f"结果：{Path(output_path).name}", font=FONT_18, fill="#9AB9C9")
    draw.rounded_rectangle((1048, 742, 1312, 785), radius=9, fill="#A61B2B", outline="#FF6075", width=2)
    draw.text((1115, 748), "强制关闭程序", font=FONT_22, fill="white")
    return canvas


def demo_frame():
    image = Image.new("RGB", (960, 540), "#D7D3C9")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 55, 895, 500), fill="#EFECE5", outline="#B7A985", width=6)
    samples = [
        UiDetection(0, .93, (140, 115, 270, 235), "PASS"),
        UiDetection(4, .91, (365, 105, 500, 300), "PASS"),
        UiDetection(7, .88, (575, 135, 770, 310), "UNKNOWN"),
        UiDetection(10, .86, (245, 300, 355, 485), "UNKNOWN"),
        UiDetection(12, .90, (610, 350, 735, 470), "PASS"),
        UiDetection(16, .95, (80, 55, 895, 500), "PASS"),
    ]
    return image, samples


class CompetitionUi:
    def __init__(self, args):
        import tkinter as tk

        self.tk = tk
        self.args = args
        self.root = tk.Tk()
        self.root.title("国赛 3D 视觉识别")
        self.root.configure(bg="#061523")
        self.root.attributes("-fullscreen", bool(args.fullscreen))
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("q", lambda _event: self.root.destroy())
        self.label = tk.Label(self.root, bg="#061523")
        self.label.pack(fill="both", expand=True)
        button_image = Image.new("RGB", (240, 46), "#A61B2B")
        ImageDraw.Draw(button_image).text((120, 23), "强制关闭程序",
                                         font=FONT_22, fill="white", anchor="mm")
        self.close_photo = ImageTk.PhotoImage(button_image)
        self.close = tk.Button(self.root, image=self.close_photo, command=self.root.destroy,
                               bg="#A61B2B", borderwidth=0)
        self.close.place(relx=.86, rely=.95, anchor="center", width=240, height=46)
        self.start = time.monotonic()
        self.last_sequence = None
        self.photo = None

    def external_status(self):
        if not self.args.status_file:
            return ""
        try:
            return self.args.status_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def update(self):
        try:
            if self.args.demo:
                frame, detections = demo_frame()
                status = "识别中"
            else:
                frame = Image.open(self.args.frame).convert("RGB")
                sequence, detections = read_worker_csv(self.args.result_csv)
                status = "识别中" if sequence is not None else (self.external_status() or "等待模型")
                self.last_sequence = sequence
            rendered = render_dashboard(
                frame, detections, table=self.args.table, status=status,
                elapsed=time.monotonic() - self.start, model_version=self.args.model_version,
                output_path=str(self.args.output_path or ""),
            )
            rendered.thumbnail((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
            self.photo = ImageTk.PhotoImage(rendered)
            self.label.configure(image=self.photo)
        except (FileNotFoundError, OSError):
            waiting = Image.new("RGB", (960, 540), "#10283A")
            wait_draw = ImageDraw.Draw(waiting)
            phase = self.external_status() or "正在准备国赛识别"
            wait_draw.text((190, 220), phase, font=FONT_36, fill="#EAF8FF")
            rendered = render_dashboard(
                waiting, [], table=self.args.table, status=phase,
                elapsed=time.monotonic() - self.start, model_version=self.args.model_version,
                output_path=str(self.args.output_path or ""),
            )
            rendered.thumbnail((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
            self.photo = ImageTk.PhotoImage(rendered)
            self.label.configure(image=self.photo)
        self.root.after(100, self.update)

    def run(self):
        self.update()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, default=Path("/dev/shm/astra_yolo_frame.jpg"))
    parser.add_argument("--result-csv", type=Path, default=Path("/dev/shm/astra_yolo_result.txt"))
    parser.add_argument("--table", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--model-version", default="national_A_20260927")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.screenshot:
        frame, detections = demo_frame()
        rendered = render_dashboard(frame, detections, table=args.table, status="识别中",
                                    elapsed=12.3, model_version=args.model_version,
                                    output_path=str(args.output_path or "CSU-3DVISION-R1.txt"))
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(args.screenshot)
        return
    CompetitionUi(args).run()


if __name__ == "__main__":
    main()
