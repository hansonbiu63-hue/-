from __future__ import annotations

import html
import json
import math
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(r"F:\UnityProgram\test2")
TEMPLATE = Path(r"C:\Users\Lenovo\Desktop\VR数码产品宣传-品牌介绍.pptx")
OUT_DIR = Path(r"C:\Users\Lenovo\Desktop\意图识别项目")
OUT_PPTX = OUT_DIR / "意图识别项目汇报_参考VR模板版_2026-05-10.pptx"
ASSET_DIR = ROOT / "Temp" / "template_style_report_assets"
MEDIA_DIR = ASSET_DIR / "template_media"
SLIDE_DIR = ASSET_DIR / "slides"
STATS_PATH = ROOT / "Assets" / "Resources" / "eye_transformer_stats.json"
ONNX_PATH = ROOT / "Assets" / "Models" / "eye_transformer_finetuned.onnx"

W, H = 1600, 900
EMU_W, EMU_H = 12192000, 6858000

NAVY = (13, 25, 42)
INK = (25, 33, 43)
MUTED = (92, 105, 119)
BLUE = (39, 105, 210)
CYAN = (0, 187, 212)
GOLD = (222, 174, 75)
PINK = (220, 57, 120)
PALE = (244, 248, 252)
WHITE = (255, 255, 255)
LINE = (205, 217, 229)
CARD = (255, 255, 255)
SOFT_BLUE = (232, 241, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf") if bold else Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(58, True)
F_H1 = font(42, True)
F_H2 = font(30, True)
F_BODY = font(23)
F_BODY_B = font(23, True)
F_SMALL = font(18)
F_SMALL_B = font(18, True)
F_TINY = font(15)
F_NUM = font(52, True)


def read_stats() -> dict:
    with STATS_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    stats["onnx_size_kb"] = round(ONNX_PATH.stat().st_size / 1024, 1) if ONNX_PATH.exists() else 0
    stats["active_feature_count"] = len(stats.get("active_feature_names", []))
    return stats


def extract_template_media() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TEMPLATE) as z:
        for name in z.namelist():
            if name.startswith("ppt/media/") and not name.endswith("/"):
                target = MEDIA_DIR / Path(name).name
                if not target.exists():
                    target.write_bytes(z.read(name))


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    fnt: ImageFont.FreeTypeFont,
    fill=INK,
    line_spacing: int = 8,
    bullet: bool = False,
) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        cur = ""
        for ch in paragraph:
            test = cur + ch
            if draw.textlength(test, font=fnt) <= width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    for idx, line in enumerate(lines):
        prefix = "• " if bullet and idx == 0 else ("  " if bullet else "")
        draw.text((x, y), prefix + line, font=fnt, fill=fill)
        y += fnt.size + line_spacing
    return y


def crop_cover(img: Image.Image, box: tuple[int, int], opacity=1.0) -> Image.Image:
    bw, bh = box
    src = img.convert("RGB")
    scale = max(bw / src.width, bh / src.height)
    resized = src.resize((int(src.width * scale), int(src.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - bw) // 2
    top = (resized.height - bh) // 2
    out = resized.crop((left, top, left + bw, top + bh))
    if opacity < 1:
        overlay = Image.new("RGB", (bw, bh), NAVY)
        out = Image.blend(overlay, out, opacity)
    return out


def draw_gradient(bg: Image.Image, start=NAVY, end=(5, 54, 72)) -> None:
    pix = bg.load()
    for y in range(bg.height):
        t = y / bg.height
        for x in range(bg.width):
            u = x / bg.width
            k = (t * 0.55 + u * 0.45)
            color = tuple(int(start[i] * (1 - k) + end[i] * k) for i in range(3))
            pix[x, y] = color


def card(draw: ImageDraw.ImageDraw, box, radius=18, fill=CARD, outline=LINE, width=2, shadow=True):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def base_slide(title: str, section: str = "", dark: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), NAVY if dark else PALE)
    draw = ImageDraw.Draw(img)
    if dark:
        draw_gradient(img)
        draw = ImageDraw.Draw(img)
        draw.line((84, 118, 1510, 118), fill=(64, 122, 154), width=2)
        draw.text((86, 58), title, font=F_H1, fill=WHITE)
        if section:
            draw.text((1320, 72), section, font=F_SMALL, fill=(176, 204, 220), anchor="ra")
    else:
        draw.rectangle((0, 0, W, 122), fill=WHITE)
        draw.rectangle((0, 0, W, 8), fill=CYAN)
        draw.text((86, 50), title, font=F_H1, fill=NAVY)
        if section:
            draw.text((1480, 62), section, font=F_SMALL, fill=BLUE, anchor="ra")
        draw.line((86, 122, 1512, 122), fill=LINE, width=2)
    draw.text((86, 846), "交互组合仿真子系统 | Unity + PICO + Eye Intention Recognition", font=F_TINY, fill=(122, 137, 150) if not dark else (164, 190, 204))
    draw.text((1512, 846), "2026-05-10", font=F_TINY, fill=(122, 137, 150) if not dark else (164, 190, 204), anchor="ra")
    return img, draw


def draw_metric(draw, x, y, value, label, color=CYAN):
    card(draw, (x, y, x + 220, y + 134), radius=18)
    draw.text((x + 110, y + 30), value, font=F_NUM, fill=color, anchor="ma")
    draw.text((x + 110, y + 92), label, font=F_SMALL, fill=MUTED, anchor="ma")


def draw_table(draw, x, y, widths, rows, row_h=58, header=True):
    cur_y = y
    for r, row in enumerate(rows):
        cur_x = x
        fill = NAVY if r == 0 and header else (WHITE if r % 2 else (248, 251, 254))
        text_fill = WHITE if r == 0 and header else INK
        for c, cell in enumerate(row):
            draw.rounded_rectangle((cur_x, cur_y, cur_x + widths[c], cur_y + row_h), radius=6, fill=fill, outline=LINE, width=1)
            draw_wrapped(draw, str(cell), (cur_x + 16, cur_y + 14), widths[c] - 28, F_SMALL_B if r == 0 and header else F_TINY, text_fill, 2)
            cur_x += widths[c]
        cur_y += row_h


def arrow(draw, start, end, fill=BLUE, width=4):
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    pts = [
        end,
        (end[0] - size * math.cos(ang - 0.45), end[1] - size * math.sin(ang - 0.45)),
        (end[0] - size * math.cos(ang + 0.45), end[1] - size * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def save_slide(img: Image.Image, idx: int) -> Path:
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    path = SLIDE_DIR / f"slide_{idx:02d}.png"
    img.save(path, "PNG")
    return path


def make_pico_dimensions() -> Path:
    img = Image.new("RGB", (1300, 520), WHITE)
    d = ImageDraw.Draw(img)
    line = (151, 166, 181)
    blue = (71, 116, 181)
    d.text((38, 24), "PICO 4 Pro 结构尺寸示意", font=font(34, True), fill=NAVY)
    d.text((38, 68), "参考用户提供的产品尺寸图重绘，用于说明硬件集成对象", font=font(20), fill=MUTED)

    # Front headset
    d.rounded_rectangle((92, 156, 444, 330), radius=85, fill=(234, 238, 241), outline=(185, 194, 204), width=4)
    d.rounded_rectangle((122, 176, 414, 294), radius=60, fill=(219, 218, 206), outline=(195, 196, 189), width=3)
    d.ellipse((261, 230, 271, 240), fill=(28, 37, 48))
    d.rounded_rectangle((60, 206, 106, 286), radius=12, fill=(229, 233, 238), outline=(180, 190, 200), width=3)
    d.rounded_rectangle((430, 206, 476, 286), radius=12, fill=(229, 233, 238), outline=(180, 190, 200), width=3)
    d.line((82, 386, 464, 386), fill=line, width=3)
    d.line((82, 370, 82, 402), fill=line, width=3)
    d.line((464, 370, 464, 402), fill=line, width=3)
    d.text((273, 410), "163mm", font=font(21, True), fill=(82, 91, 102), anchor="ma")

    # Side headset and strap
    d.rounded_rectangle((610, 170, 728, 320), radius=28, fill=(233, 237, 242), outline=(184, 196, 207), width=4)
    d.polygon([(704, 170), (790, 126), (852, 198), (830, 322), (730, 320)], fill=(130, 133, 135))
    d.rounded_rectangle((716, 205, 1112, 255), radius=24, fill=(232, 236, 241), outline=(190, 199, 209), width=3)
    d.rounded_rectangle((1110, 184, 1226, 276), radius=45, fill=(232, 236, 241), outline=(190, 199, 209), width=3)
    d.line((580, 170, 580, 320), fill=line, width=3)
    d.line((565, 170, 595, 170), fill=line, width=3)
    d.line((565, 320, 595, 320), fill=line, width=3)
    d.text((540, 246), "80mm", font=font(21, True), fill=(82, 91, 102), anchor="ra")
    d.line((643, 116, 720, 116), fill=line, width=3)
    d.line((643, 100, 643, 132), fill=line, width=3)
    d.line((720, 100, 720, 132), fill=line, width=3)
    d.text((682, 82), "35.8mm", font=font(21, True), fill=(82, 91, 102), anchor="ma")
    d.line((612, 386, 1215, 386), fill=line, width=3)
    d.line((612, 370, 612, 402), fill=line, width=3)
    d.line((1215, 370, 1215, 402), fill=line, width=3)
    d.text((914, 410), "255 - 310mm", font=font(21, True), fill=(82, 91, 102), anchor="ma")

    path = ASSET_DIR / "pico_dimensions.png"
    img.save(path)
    return path


def make_experiment_flow() -> Path:
    img = Image.new("RGB", (1300, 700), WHITE)
    d = ImageDraw.Draw(img)
    d.text((42, 28), "实验流程与数据闭环", font=font(34, True), fill=NAVY)
    d.text((42, 72), "根据用户提供流程图重绘：任务选择分支 + 数据预处理 + 标签对齐 + 训练评估", font=font(20), fill=MUTED)
    outline = (68, 123, 196)
    fill = (248, 252, 255)

    def node(x, y, text, w=128, h=74):
        d.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=4)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            d.text((x + w / 2, y + 20 + i * 26), line, font=font(24, True), fill=outline, anchor="ma")
        return (x, y, x + w, y + h)

    boxes = {}
    top_y = 230
    labels = ["实验开始", "环境控制", "设备校准", "任务选择", "数据采集", "数据预处理"]
    xs = [38, 210, 382, 554, 780, 1010]
    for x, lab in zip(xs, labels):
        boxes[lab] = node(x, top_y, lab, 128 if lab != "数据预处理" else 150)
    boxes["静态任务"] = node(554, 108, "静态任务")
    boxes["动态任务"] = node(554, 360, "动态任务")
    boxes["标签对齐"] = node(1060, 542, "标签对齐")
    boxes["数据集划分"] = node(830, 542, "数据集\n划分")
    boxes["模型训练"] = node(600, 542, "模型训练")
    boxes["性能评估"] = node(370, 542, "性能评估")
    boxes["结束"] = node(140, 542, "结束")

    for a, b in zip(labels[:-1], labels[1:]):
        x1, y1, x2, y2 = boxes[a]
        nx1, ny1, nx2, ny2 = boxes[b]
        arrow(d, (x2, (y1 + y2) // 2), (nx1, (ny1 + ny2) // 2), outline, 4)
    # Task branch arrows
    x1, y1, x2, y2 = boxes["任务选择"]
    sx1, sy1, sx2, sy2 = boxes["静态任务"]
    dx1, dy1, dx2, dy2 = boxes["动态任务"]
    arrow(d, ((x1 + x2) // 2, y1), ((sx1 + sx2) // 2, sy2), outline, 4)
    arrow(d, ((x1 + x2) // 2, y2), ((dx1 + dx2) // 2, dy1), outline, 4)
    px1, py1, px2, py2 = boxes["数据预处理"]
    lx1, ly1, lx2, ly2 = boxes["标签对齐"]
    arrow(d, ((px1 + px2) // 2, py2), ((px1 + px2) // 2, ly1 + 10), outline, 4)
    arrow(d, ((px1 + px2) // 2, ly1 + 10), ((lx1 + lx2) // 2, ly1), outline, 4)
    bottom = ["标签对齐", "数据集划分", "模型训练", "性能评估", "结束"]
    for a, b in zip(bottom[:-1], bottom[1:]):
        x1, y1, x2, y2 = boxes[a]
        nx1, ny1, nx2, ny2 = boxes[b]
        arrow(d, (x1, (y1 + y2) // 2), (nx2, (ny1 + ny2) // 2), outline, 4)

    path = ASSET_DIR / "experiment_flow.png"
    img.save(path)
    return path


def make_uml_architecture() -> Path:
    img = Image.new("RGB", (1400, 760), WHITE)
    d = ImageDraw.Draw(img)
    d.text((44, 28), "UML 软件架构图（组件/分层视图）", font=font(34, True), fill=NAVY)
    d.text((44, 72), "运行时只推理：训练脚本离线产出 ONNX 与 stats，Unity/PICO 端加载模型并输出推荐结果", font=font(20), fill=MUTED)

    def package(x, y, w, h, title, items, color):
        d.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=(248, 252, 255), outline=color, width=4)
        d.rectangle((x, y, x + w, y + 48), fill=color)
        d.text((x + 18, y + 11), title, font=font(22, True), fill=WHITE)
        yy = y + 70
        for item in items:
            d.rounded_rectangle((x + 22, yy, x + w - 22, yy + 44), radius=8, fill=WHITE, outline=LINE, width=2)
            draw_wrapped(d, f"«component» {item}", (x + 38, yy + 9), w - 78, font(15), INK, 1)
            yy += 58

    package(56, 124, 372, 190, "设备与接入层", ["PICO 4 Pro", "PXR SDK", "XR Interaction Toolkit"], BLUE)
    package(514, 124, 372, 190, "任务与交互层", ["TaskManager", "ControlItem", "ConfirmInput"], CYAN)
    package(972, 124, 372, 190, "UI与反馈层", ["VRDashboardPolish", "World-Space HUD", "Top3 Recommend HUD"], PINK)
    package(56, 380, 372, 190, "数据采集层", ["EyeTrackingManager", "GazeDataLogger", "Frame/Summary CSV"], BLUE)
    package(514, 380, 372, 190, "算法推理层", ["RealtimeRecommender", "OnnxInference", "TinyTransformer ONNX"], CYAN)
    package(972, 380, 372, 190, "评估与训练层", ["WindowEvaluator", "train_transformer.py", "Stats JSON"], GOLD)

    arrow(d, (428, 220), (514, 220), BLUE, 4)
    arrow(d, (886, 220), (972, 220), CYAN, 4)
    arrow(d, (242, 314), (242, 380), BLUE, 4)
    arrow(d, (428, 476), (514, 476), BLUE, 4)
    arrow(d, (886, 476), (972, 476), GOLD, 4)
    arrow(d, (700, 380), (700, 314), CYAN, 4)
    arrow(d, (972, 252), (886, 446), PINK, 4)
    d.rounded_rectangle((238, 626, 526, 690), radius=12, fill=(241, 248, 255), outline=BLUE, width=3)
    d.text((382, 647), "Unity Frame CSV 数据集", font=font(20, True), fill=BLUE, anchor="ma")
    d.rounded_rectangle((796, 626, 1084, 690), radius=12, fill=(255, 249, 237), outline=GOLD, width=3)
    d.text((940, 647), "GazeBaseVR 预训练数据", font=font(20, True), fill=(158, 112, 24), anchor="ma")
    arrow(d, (382, 626), (600, 570), BLUE, 4)
    arrow(d, (940, 626), (1010, 570), GOLD, 4)

    path = ASSET_DIR / "uml_architecture.png"
    img.save(path)
    return path


def paste_image(frame: Image.Image, path: Path, box, radius=0, shadow=False, fit=False, bg=WHITE):
    x1, y1, x2, y2 = box
    src = Image.open(path).convert("RGB")
    bw, bh = x2 - x1, y2 - y1
    if fit:
        crop = Image.new("RGB", (bw, bh), bg)
        fitted = src.copy()
        fitted.thumbnail((bw - 8, bh - 8), Image.Resampling.LANCZOS)
        crop.paste(fitted, ((bw - fitted.width) // 2, (bh - fitted.height) // 2))
    else:
        crop = crop_cover(src, (bw, bh), 1.0)
    if radius:
        mask = Image.new("L", (bw, bh), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, bw, bh), radius=radius, fill=255)
        if shadow:
            sh = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            sd.rounded_rectangle((x1 + 12, y1 + 14, x2 + 12, y2 + 14), radius=radius, fill=(0, 0, 0, 50))
            frame.paste(sh.convert("RGB"), mask=sh.split()[-1])
        frame.paste(crop, (x1, y1), mask)
    else:
        frame.paste(crop, (x1, y1))


def create_slides(stats: dict) -> list[Path]:
    extract_template_media()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    pico_img = make_pico_dimensions()
    flow_img = make_experiment_flow()
    uml_img = make_uml_architecture()
    vr_cover = MEDIA_DIR / "image2.png"
    vr_user = MEDIA_DIR / "image10.png"
    business = MEDIA_DIR / "image6.jpeg"
    building = MEDIA_DIR / "image4.jpeg"

    slides: list[Path] = []

    # 1 Cover
    img = Image.new("RGB", (W, H), NAVY)
    draw_gradient(img, (8, 16, 31), (0, 103, 120))
    if vr_cover.exists():
        paste_image(img, vr_cover, (1040, 0, 1600, 900), radius=0)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle((940, 0, 1600, 900), fill=(3, 20, 36, 78))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 15, H), fill=CYAN)
    d.text((102, 112), "基于眼动追踪的\n意图识别项目", font=F_TITLE, fill=WHITE)
    d.text((106, 278), "交互组合仿真子系统研发进度与阶段性成果汇报", font=F_H2, fill=(212, 232, 242))
    d.text((108, 338), "Unity + PICO 4 Pro | 规则推理 + 递归贝叶斯 + TinyEyeTransformer", font=F_BODY, fill=(184, 215, 225))
    for i, (value, label, color) in enumerate([("47", "输入维度", CYAN), ("150", "窗口帧", GOLD), ("10", "AOI类别", CYAN), ("ONNX", "离线推理", GOLD)]):
        x = 104 + i * 205
        d.rounded_rectangle((x, 610, x + 170, 738), radius=16, fill=(255, 255, 255, 32), outline=(120, 178, 198), width=2)
        d.text((x + 85, 634), value, font=font(42, True), fill=color, anchor="ma")
        d.text((x + 85, 692), label, font=F_SMALL, fill=(210, 230, 238), anchor="ma")
    d.text((108, 794), "日期：2026年5月10日", font=F_SMALL, fill=(183, 209, 222))
    slides.append(save_slide(img, 1))

    # 2 Progress alignment
    img, d = base_slide("01 项目进度对齐", "Progress")
    draw_table(d, 86, 164, [260, 300, 730], [
        ["模块", "当前状态", "结论"],
        ["PICO / Unity 平台", "已完成", "眼动采集、XR交互、AOI映射和任务流程已具备运行基础。"],
        ["静态/动态任务", "已完成", "支持目标随机、动态释放、干扰控件和可复现实验种子。"],
        ["规则法 / 贝叶斯法", "已完成", "RealtimeIntentionRecommender 已实现 Top3、置信度和窗口级输出。"],
        ["Transformer", "已集成", "ONNX/Sentis 推理壳已接入，训练链路迁移到离线脚本。"],
        ["数据集与验收", "进行中", "需补齐10名被试、100次实验、20小时数据和正式指标报告。"],
    ], row_h=72)
    d.text((92, 720), "阶段判断：工程已进入系统集成与验证阶段，后续重点从“功能是否可跑”转为“数据规模、模型复训、真机稳定性与验收证据”。", font=F_BODY_B, fill=NAVY)
    slides.append(save_slide(img, 2))

    # 3 PICO dimensions
    img, d = base_slide("02 PICO 硬件对象与尺寸说明", "Hardware")
    card(d, (70, 156, 1530, 734), radius=24)
    paste_image(img, pico_img, (118, 194, 1482, 698), radius=14, fit=True)
    d.text((96, 760), "用途：明确仿真系统的头显载体、佩戴结构和空间约束，为眼动校准、UI距离、视野布局和PICO说明书章节提供图示依据。", font=F_SMALL_B, fill=INK)
    slides.append(save_slide(img, 3))

    # 4 Experiment flow
    img, d = base_slide("03 实验流程与数据闭环", "Workflow")
    card(d, (74, 156, 1526, 770), radius=24)
    paste_image(img, flow_img, (118, 190, 1482, 724), radius=14, fit=True)
    slides.append(save_slide(img, 4))

    # 5 UML architecture
    img, d = base_slide("04 UML 软件架构图", "Architecture")
    card(d, (62, 144, 1538, 794), radius=24)
    paste_image(img, uml_img, (94, 176, 1506, 756), radius=14, fit=True)
    slides.append(save_slide(img, 5))

    # 6 System architecture narrative
    img, d = base_slide("05 系统运行架构", "System")
    flow = [
        ("PICO 4 Pro", "眼动追踪\n头显姿态\n手柄确认"),
        ("Unity采集层", "PXR SDK\n坐标映射\nAOI命中"),
        ("数据记录层", "Frame CSV\nSummary CSV\nEval CSV"),
        ("算法推理层", "规则法\n递归贝叶斯\nTransformer"),
        ("推荐反馈层", "Top1/Top3\n置信度\n确认状态"),
    ]
    for i, (title, body) in enumerate(flow):
        x = 74 + i * 300
        card(d, (x, 210, x + 224, 432), radius=20)
        d.text((x + 24, 238), title, font=F_BODY_B, fill=NAVY)
        draw_wrapped(d, body, (x + 24, 292), 170, F_SMALL, MUTED, 8)
        if i < len(flow) - 1:
            arrow(d, (x + 230, 320), (x + 286, 320), CYAN, 5)
    card(d, (94, 550, 1488, 700), radius=20, fill=(247, 252, 255), outline=(193, 222, 235))
    d.text((124, 582), "核心原则", font=F_BODY_B, fill=BLUE)
    draw_wrapped(d, "Unity 端只做采集、推理、交互与评估；Transformer 不在运行时训练。模型由离线脚本预训练/微调后导出 ONNX 与 stats，再由 Sentis 加载执行。", (124, 626), 1280, F_BODY, INK)
    slides.append(save_slide(img, 6))

    # 7 Data structure
    img, d = base_slide("06 当前数据结构", "Data")
    draw_table(d, 78, 160, [270, 510, 670], [
        ["类别", "核心字段", "用途"],
        ["会话信息", "subject_id, session_id, round_index, mode, phase", "标识被试、实验轮次、静态/动态任务和任务阶段。"],
        ["时间索引", "timestamp_ms, frame_idx", "用于滑动窗口切分、延迟计算和算法对齐。"],
        ["眼动几何", "gaze_origin_*, gaze_dir_*, gaze_point_*", "描述3D视线、投射点和空间方向。"],
        ["屏幕映射", "screen_x_norm, screen_y_norm, hit_ui_plane", "映射到世界空间UI平面，判断是否命中。"],
        ["AOI标签", "aoi_id, aoi_valid, aoi_name, target_controls", "控件0-9、多目标监督标签与命中有效性。"],
        ["生理/有效性", "pupil_*, openness_*, tracking_valid", "眨眼、瞳孔缺失、无效帧容错与质量控制。"],
    ], row_h=67)
    slides.append(save_slide(img, 7))

    # 8 Transformer parameters
    img, d = base_slide("07 Transformer 模型参数与数据维度", "Transformer")
    values = [
        (str(stats.get("feature_dim", 47)), "feature_dim"),
        (str(stats.get("window_frames", 150)), "window_frames"),
        (str(stats.get("step_frames", 15)), "step_frames"),
        (str(stats.get("aoi_count", 10)), "AOI classes"),
        (str(stats.get("threshold", 0.3)), "threshold"),
    ]
    for i, (v, lab) in enumerate(values):
        draw_metric(d, 86 + i * 292, 164, v, lab, CYAN if i % 2 == 0 else GOLD)
    draw_table(d, 92, 360, [330, 430, 720], [
        ["项目", "当前配置", "说明"],
        ["输入张量", "[1, 150, 47]", "单批次、3秒窗口、47维眼动与AOI特征。"],
        ["输出形式", "10维 sigmoid 概率", "多标签意图识别，对应控件0-9。"],
        ["训练模式", "pretrain + finetune", "GazeBaseVR预训练，Unity Frame CSV微调。"],
        ["Unity部署", "eye_transformer_finetuned.onnx + stats.json", "运行时只加载模型推理，不训练。"],
        ["有效帧策略", f"min_valid_frames={stats.get('min_valid_frames', 20)}", "低质量/眨眼窗口进入等待或低置信状态。"],
    ], row_h=62)
    d.text((98, 732), f"当前模型文件：Assets/Models/eye_transformer_finetuned.onnx，约 {stats.get('onnx_size_kb', 0)} KB；stats stage={stats.get('stage', 'finetune')}。", font=F_SMALL_B, fill=NAVY)
    slides.append(save_slide(img, 8))

    # 9 Offline training pipeline
    img, d = base_slide("08 Transformer 离线训练与部署流程", "Pipeline")
    steps = [
        ("GazeBaseVR", "原始CSV\n250Hz→50Hz"),
        ("预训练", "学习通用眼动\n时序表征"),
        ("Unity Frame CSV", "10控件标签\n0-9多目标"),
        ("微调", "Focal Loss\nWeighted GAP"),
        ("导出", "ONNX\nstats JSON"),
        ("Unity推理", "Sentis加载\nTop3输出"),
    ]
    for i, (title, body) in enumerate(steps):
        x = 70 + i * 246
        card(d, (x, 220, x + 182, 420), radius=18)
        d.ellipse((x + 62, 178, x + 120, 236), fill=CYAN if i % 2 == 0 else GOLD)
        d.text((x + 91, 189), str(i + 1), font=F_SMALL_B, fill=WHITE, anchor="ma")
        d.text((x + 91, 260), title, font=F_SMALL_B, fill=NAVY, anchor="ma")
        draw_wrapped(d, body, (x + 26, 312), 135, F_TINY, MUTED, 5)
        if i < len(steps) - 1:
            arrow(d, (x + 190, 320), (x + 238, 320), BLUE, 4)
    card(d, (106, 558, 1470, 690), radius=18, fill=(248, 252, 255), outline=(192, 220, 234))
    draw_wrapped(d, "CLI 产物：eye_transformer_pretrained.pt、eye_transformer_finetuned.pt、Assets/Models/eye_transformer_finetuned.onnx、Assets/Resources/eye_transformer_stats.json、训练/验证/测试指标CSV。", (138, 590), 1300, F_BODY, INK)
    slides.append(save_slide(img, 9))

    # 10 Algorithms comparison
    img, d = base_slide("09 三算法对比与验收指标", "Evaluation")
    draw_table(d, 80, 160, [260, 360, 360, 450], [
        ["算法", "输入", "输出", "当前情况"],
        ["规则法", "AOI命中、注视时长、瞳孔有效性", "Top3 / 命中计数", "已实现，适合静态任务和可解释验收。"],
        ["惩罚加权递归贝叶斯", "注视距离、速度、稳定性、瞳孔Z分数", "后验概率 / 置信度", "已实现，适合动态噪声下平滑判断。"],
        ["TinyEyeTransformer", "150x47时序窗口", "10维多标签概率", "ONNX推理已集成，正式指标需基于完整数据复训。"],
    ], row_h=86)
    card(d, (100, 600, 450, 720), radius=16)
    d.text((128, 628), "静态任务目标", font=F_BODY_B, fill=BLUE)
    d.text((128, 676), "三算法准确率均 ≥ 90%", font=F_H2, fill=NAVY)
    card(d, (590, 600, 940, 720), radius=16)
    d.text((618, 628), "动态任务目标", font=F_BODY_B, fill=BLUE)
    d.text((618, 676), "至少一种算法 ≥ 85%", font=F_H2, fill=NAVY)
    card(d, (1080, 600, 1430, 720), radius=16)
    d.text((1108, 628), "补齐材料", font=F_BODY_B, fill=BLUE)
    d.text((1108, 676), "正式CSV与报告", font=F_H2, fill=NAVY)
    slides.append(save_slide(img, 10))

    # 11 UI and interaction
    img, d = base_slide("10 界面高级化与自然交互方案", "UI/UX")
    if business.exists():
        paste_image(img, business, (1010, 142, 1540, 770), radius=22, shadow=True)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((1010, 142, 1540, 770), radius=22, fill=(0, 0, 0, 82))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(img)
    items = [
        ("世界空间仪表盘", "低眩光深色底、TextMesh Pro、统一字号层级、模块化状态区。"),
        ("眼动预测 + 手柄确认", "注视只表达候选意图，最终操作由确认键触发，降低误触。"),
        ("渐进式反馈", "300-500ms轻反馈，800ms稳定注视，Top3候选和置信度渐变。"),
        ("沉浸式推荐层", "任务目标、注视状态、推荐结果、确认状态、实验状态五类信息分区。"),
    ]
    for i, (title, body) in enumerate(items):
        y = 166 + i * 138
        card(d, (88, y, 900, y + 100), radius=18)
        d.rectangle((88, y, 98, y + 100), fill=CYAN if i % 2 == 0 else GOLD)
        d.text((124, y + 18), title, font=F_BODY_B, fill=NAVY)
        draw_wrapped(d, body, (124, y + 52), 720, F_SMALL, MUTED, 3)
    d.text((1042, 654), "目标：从实验样机界面升级为可交付的VR监控工作台体验", font=F_SMALL_B, fill=WHITE)
    slides.append(save_slide(img, 11))

    # 12 Progress and next steps
    img, d = base_slide("11 项目进度总结与下一步", "Next")
    if building.exists():
        paste_image(img, building, (0, 122, 1600, 900), radius=0)
        overlay = Image.new("RGBA", (W, H), (245, 249, 252, 222))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, W, 122), fill=WHITE)
        d.rectangle((0, 0, W, 8), fill=CYAN)
        d.text((86, 50), "11 项目进度总结与下一步", font=F_H1, fill=NAVY)
        d.line((86, 122, 1512, 122), fill=LINE, width=2)
    cols = [
        ("已完成", ["PICO/XRI平台与眼动采集", "静态/动态任务流程", "规则法与递归贝叶斯", "ONNX推理壳与评估导出"]),
        ("进行中", ["正式数据集补齐", "GazeBaseVR预训练复现", "Unity数据微调与消融", "PICO真机验收截图/CSV"]),
        ("交付物", ["模板版PPT", "软件说明书", "算法对比报告", "模型/脚本/数据质检报告"]),
    ]
    for i, (title, bullets) in enumerate(cols):
        x = 86 + i * 500
        card(d, (x, 192, x + 420, 652), radius=22)
        d.text((x + 32, 230), title, font=F_H2, fill=BLUE if i != 1 else GOLD)
        y = 306
        for b in bullets:
            d.ellipse((x + 32, y + 8, x + 45, y + 21), fill=CYAN if i != 1 else GOLD)
            draw_wrapped(d, b, (x + 62, y), 320, F_BODY, INK, 5)
            y += 76
    d.text((102, 742), "收口判断：项目基础能力已完成，最终验收需要以完整被试数据、模型指标和真机运行证据形成闭环。", font=F_BODY_B, fill=NAVY)
    slides.append(save_slide(img, 12))

    return slides


def rels_xml(rels: list[tuple[str, str, str]]) -> str:
    body = "\n".join(
        f'<Relationship Id="{esc(rid)}" Type="{esc(typ)}" Target="{esc(target)}"/>'
        for rid, typ, target in rels
    )
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n{body}\n</Relationships>'


def slide_xml(image_rid: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm></p:grpSpPr>
      <p:pic>
        <p:nvPicPr><p:cNvPr id="2" name="slide.png"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="{image_rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def content_types(slide_count: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
{slide_overrides}
</Types>'''


def presentation_xml(slide_count: int) -> str:
    ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, slide_count + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{EMU_W}" cy="{EMU_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="zh-CN"/></a:defPPr></p:defaultTextStyle>
</p:presentation>'''


def theme_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="VR Business Theme">
  <a:themeElements>
    <a:clrScheme name="VR"><a:dk1><a:srgbClr val="0D192A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="193044"/></a:dk2><a:lt2><a:srgbClr val="F4F8FC"/></a:lt2><a:accent1><a:srgbClr val="00BBD4"/></a:accent1><a:accent2><a:srgbClr val="2769D2"/></a:accent2><a:accent3><a:srgbClr val="DEAE4B"/></a:accent3><a:accent4><a:srgbClr val="DC3978"/></a:accent4><a:accent5><a:srgbClr val="5C6977"/></a:accent5><a:accent6><a:srgbClr val="CDD9E5"/></a:accent6><a:hlink><a:srgbClr val="2769D2"/></a:hlink><a:folHlink><a:srgbClr val="DC3978"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Microsoft YaHei"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="VR"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>'''


def slide_master_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>'''


def slide_layout_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''


def create_pptx(slides: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        pres_rels = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml")]
        for i in range(1, len(slides) + 1):
            pres_rels.append((f"rId{i+1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i}.xml"))
        pres_rels.extend([
            (f"rId{len(slides)+2}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
            (f"rId{len(slides)+3}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
            (f"rId{len(slides)+4}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
            (f"rId{len(slides)+5}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
        ])
        z.writestr("ppt/_rels/presentation.xml.rels", rels_xml(pres_rels))
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
        ]))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml"),
        ]))
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        z.writestr("ppt/presProps.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
        z.writestr("ppt/viewProps.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
        z.writestr("ppt/tableStyles.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>')
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        z.writestr("docProps/core.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>意图识别项目汇报_参考VR模板版</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>''')
        z.writestr("docProps/app.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex PPTX Renderer</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{len(slides)}</Slides></Properties>''')
        for i, slide in enumerate(slides, 1):
            z.write(slide, f"ppt/media/slide{i}.png")
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml("rId2"))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels_xml([
                ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
                ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/slide{i}.png"),
            ]))


def main() -> None:
    stats = read_stats()
    slides = create_slides(stats)
    create_pptx(slides, OUT_PPTX)
    print(OUT_PPTX)
    print(ASSET_DIR)


if __name__ == "__main__":
    main()
