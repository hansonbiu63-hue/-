from __future__ import annotations

import json
import time
from pathlib import Path

import win32com.client


ROOT = Path(r"F:\UnityProgram\test2")
OUT_DIR = Path(r"C:\Users\Lenovo\Desktop\意图识别项目")
PPT_OUT = OUT_DIR / "意图识别项目汇报_完善商务版_2026-05-10.pptx"
DOC_OUT = OUT_DIR / "交互组合仿真子系统软件说明书_2026-05-10.docx"
STATS_PATH = ROOT / "Assets" / "Resources" / "eye_transformer_stats.json"
ONNX_PATH = ROOT / "Assets" / "Models" / "eye_transformer_finetuned.onnx"


def rgb(r: int, g: int, b: int) -> int:
    return r + g * 256 + b * 65536


NAVY = rgb(18, 31, 45)
INK = rgb(31, 45, 56)
MUTED = rgb(91, 104, 114)
TEAL = rgb(0, 152, 166)
GOLD = rgb(212, 170, 84)
PALE = rgb(244, 247, 250)
WHITE = rgb(255, 255, 255)
LINE = rgb(213, 221, 229)


def read_stats() -> dict:
    with STATS_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    stats["onnx_size_kb"] = round(ONNX_PATH.stat().st_size / 1024, 1) if ONNX_PATH.exists() else 0
    stats["active_feature_count"] = len(stats.get("active_feature_names", []))
    stats["dropped_feature_text"] = "、".join(stats.get("dropped_feature_names", [])) or "无"
    return stats


def add_bg(slide, title: str, section: str = ""):
    slide.FollowMasterBackground = False
    bg = slide.Shapes.AddShape(1, 0, 0, 960, 540)
    bg.Fill.ForeColor.RGB = PALE
    bg.Line.Visible = 0
    bar = slide.Shapes.AddShape(1, 0, 0, 960, 42)
    bar.Fill.ForeColor.RGB = NAVY
    bar.Line.Visible = 0
    add_text(slide, title, 34, 64, 590, 32, size=18, bold=True, color=WHITE)
    if section:
        add_text(slide, section, 742, 64, 160, 28, size=9, color=rgb(196, 209, 220), align=3)
    add_text(slide, "交互组合仿真子系统 | 2026-05-10", 610, 504, 280, 18, size=8, color=MUTED, align=3)


def add_text(slide, text: str, x: float, y: float, w: float, h: float, size=14, bold=False, color=INK, align=1):
    box = slide.Shapes.AddTextbox(1, x, y, w, h)
    box.TextFrame.MarginLeft = 0
    box.TextFrame.MarginRight = 0
    box.TextFrame.MarginTop = 0
    box.TextFrame.MarginBottom = 0
    tr = box.TextFrame.TextRange
    tr.Text = text
    tr.Font.Name = "Microsoft YaHei"
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = color
    tr.ParagraphFormat.Alignment = align
    return box


def add_card(slide, x, y, w, h, title, body, accent=TEAL):
    s = slide.Shapes.AddShape(1, x, y, w, h)
    s.Fill.ForeColor.RGB = WHITE
    s.Line.ForeColor.RGB = LINE
    add_text(slide, title, x + 18, y + 14, w - 36, 22, size=12, bold=True, color=INK)
    a = slide.Shapes.AddShape(1, x, y, 5, h)
    a.Fill.ForeColor.RGB = accent
    a.Line.Visible = 0
    add_text(slide, body, x + 18, y + 44, w - 36, h - 56, size=10, color=MUTED)
    return s


def add_metric(slide, x, y, w, h, value, label, accent=TEAL):
    s = slide.Shapes.AddShape(1, x, y, w, h)
    s.Fill.ForeColor.RGB = WHITE
    s.Line.ForeColor.RGB = LINE
    add_text(slide, value, x + 14, y + 14, w - 28, 36, size=24, bold=True, color=accent, align=2)
    add_text(slide, label, x + 12, y + 56, w - 24, 28, size=9, color=MUTED, align=2)


def add_table(slide, x, y, w, h, headers, rows):
    table_shape = slide.Shapes.AddTable(len(rows) + 1, len(headers), x, y, w, h)
    table = table_shape.Table
    for c, header in enumerate(headers, 1):
        cell = table.Cell(1, c)
        cell.Shape.Fill.ForeColor.RGB = NAVY
        cell.Shape.TextFrame.TextRange.Text = header
        cell.Shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
        cell.Shape.TextFrame.TextRange.Font.Size = 9
        cell.Shape.TextFrame.TextRange.Font.Bold = -1
        cell.Shape.TextFrame.TextRange.Font.Color.RGB = WHITE
    for r, row in enumerate(rows, 2):
        for c, value in enumerate(row, 1):
            cell = table.Cell(r, c)
            cell.Shape.Fill.ForeColor.RGB = WHITE if r % 2 == 0 else rgb(249, 251, 253)
            cell.Shape.TextFrame.TextRange.Text = str(value)
            cell.Shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
            cell.Shape.TextFrame.TextRange.Font.Size = 8
            cell.Shape.TextFrame.TextRange.Font.Color.RGB = INK
    return table_shape


def add_flow(slide, y, items):
    x = 74
    for i, (title, body) in enumerate(items):
        add_card(slide, x, y, 160, 96, title, body, TEAL if i % 2 == 0 else GOLD)
        if i < len(items) - 1:
            add_text(slide, ">", x + 170, y + 35, 24, 28, size=18, bold=True, color=TEAL, align=2)
        x += 196


def build_ppt(stats: dict):
    ppt = win32com.client.DispatchEx("PowerPoint.Application")
    ppt.Visible = True
    ppt.DisplayAlerts = 0
    time.sleep(0.8)
    pres = ppt.Presentations.Add()
    pres.PageSetup.SlideWidth = 960
    pres.PageSetup.SlideHeight = 540

    slide = pres.Slides.Add(1, 12)
    cover = slide.Shapes.AddShape(1, 0, 0, 960, 540)
    cover.Fill.ForeColor.RGB = NAVY
    cover.Line.Visible = 0
    slide.Shapes.AddShape(1, 0, 386, 960, 154).Fill.ForeColor.RGB = rgb(10, 90, 105)
    add_text(slide, "基于眼动追踪的意图识别项目", 70, 118, 720, 48, size=30, bold=True, color=WHITE)
    add_text(slide, "交互组合仿真子系统 研发进度与阶段性成果汇报", 72, 176, 680, 30, size=15, color=rgb(215, 226, 235))
    add_text(slide, "Unity + PICO 4 Pro | 规则推理 + 递归贝叶斯 + TinyEyeTransformer", 72, 222, 700, 26, size=12, color=rgb(190, 207, 217))
    add_text(slide, "汇报人：李玉瑶\n日期：2026年5月10日", 72, 430, 240, 44, size=11, color=WHITE)
    add_metric(slide, 650, 118, 105, 88, "47", "Transformer输入维度", GOLD)
    add_metric(slide, 770, 118, 105, 88, "150", "3秒滑窗帧数", TEAL)
    add_metric(slide, 650, 226, 105, 88, "10", "AOI控件类别", GOLD)
    add_metric(slide, 770, 226, 105, 88, "ONNX", "Unity端离线推理", TEAL)

    slide = pres.Slides.Add(2, 12)
    add_bg(slide, "01 项目定位与阶段结论", "Executive Summary")
    add_card(slide, 54, 88, 258, 150, "项目目标", "面向 PICO 4 Pro 的 VR/AR 实验环境，融合眼动通道、任务上下文与显示界面状态，实时推断用户正在监控或寻找的目标控件。", TEAL)
    add_card(slide, 350, 88, 258, 150, "当前阶段", "平台采集链路、静态/动态任务、规则法、递归贝叶斯、Transformer ONNX 推理壳和评估导出均已落地。", GOLD)
    add_card(slide, 646, 88, 258, 150, "关键变化", "Transformer 已从运行时训练口径调整为离线预训练/微调后导入 Unity 推理，部署时校验 stats 与模型参数。", TEAL)
    add_table(slide, 74, 280, 812, 156, ["模块", "完成情况", "说明"], [
        ["实验平台", "已完成", "PICO 眼动采集、坐标映射、任务流程、CSV导出"],
        ["规则算法", "已完成", "3秒窗口、AOI命中、注视时长、Top3输出"],
        ["递归贝叶斯", "已完成", "后验概率、惩罚因子、动态阈值、置信度输出"],
        ["Transformer", "已集成", "ONNX推理壳已接入，离线训练脚本已归档"],
        ["数据集", "进行中", "采集规模需继续补齐至验收要求"],
    ])

    slide = pres.Slides.Add(3, 12)
    add_bg(slide, "02 系统总体架构", "Architecture")
    add_flow(slide, 92, [
        ("PICO 4 Pro", "眼动追踪\n头显姿态\n手柄确认"),
        ("Unity采集层", "PXR SDK\n坐标映射\nAOI命中"),
        ("数据记录层", "Frame CSV\nRound Summary\n评估窗口"),
        ("算法层", "规则\n递归贝叶斯\nTransformer"),
    ])
    add_flow(slide, 250, [
        ("任务系统", "静态任务\n动态任务\n随机释放"),
        ("推荐反馈", "Top1/Top3\n置信度\n低眩光UI"),
        ("评估导出", "Accuracy\nHit@3\nSubset"),
        ("交付物", "源码\n模型\n报告"),
    ])

    slide = pres.Slides.Add(4, 12)
    add_bg(slide, "03 PICO硬件集成与实验场景", "Platform")
    add_card(slide, 54, 86, 250, 156, "硬件与运行环境", "PICO 4 Pro 作为眼动采集与沉浸显示设备；Windows 主机用于 Unity 编辑、模型训练、数据归档与汇总分析。", TEAL)
    add_card(slide, 354, 86, 250, 156, "眼动采集链路", "通过 PXR 接口读取融合视线、眼动方向、瞳孔/睁眼状态，并投射至世界空间 UI 平面完成 AOI 命中。", GOLD)
    add_card(slide, 654, 86, 250, 156, "交互方式", "注视只表达意图候选，确认由手柄完成；系统降低误触风险，保留用户主动确认权。", TEAL)
    add_table(slide, 90, 280, 780, 132, ["项目", "设置"], [
        ["设备", "PICO 4 Pro / PICO Unity Integration SDK"],
        ["引擎", "Unity 2022.3 + XR Interaction Toolkit + Sentis"],
        ["采样与窗口", "50Hz采样口径，3秒/150帧滑动窗口"],
        ["任务界面", "10个AOI控件，编号0-9，目标数量随机"],
    ])

    slide = pres.Slides.Add(5, 12)
    add_bg(slide, "04 实验任务与自然交互逻辑", "Interaction")
    add_card(slide, 58, 92, 250, 130, "静态任务", "固定时间窗内发布一组目标控件，验证稳定注视与多目标识别准确性。", TEAL)
    add_card(slide, 356, 92, 250, 130, "动态任务", "任务释放时间、目标数量和干扰控件随机变化，模拟高认知负荷监控。", GOLD)
    add_card(slide, 654, 92, 250, 130, "人因策略", "300-500ms轻反馈、800ms稳定注视判断、手柄二次确认，减少误触与视觉疲劳。", TEAL)
    add_flow(slide, 282, [
        ("自由浏览", "记录背景眼动\n等待任务释放"),
        ("目标派发", "显示目标控件\n更新标签"),
        ("注视积累", "值增长\n符号保持"),
        ("手柄确认", "完成任务\n进入下一轮"),
    ])

    slide = pres.Slides.Add(6, 12)
    add_bg(slide, "05 数据结构与记录字段", "Data")
    add_table(slide, 52, 88, 856, 252, ["字段类别", "核心字段", "用途"], [
        ["会话信息", "subject_id, session_id, round_index, mode, phase", "定位被试、实验轮次和任务阶段"],
        ["时间索引", "timestamp_ms, frame_idx", "窗口切分、延迟计算、时序对齐"],
        ["空间视线", "gaze_origin, gaze_dir, gaze_point", "三维视线投射与轨迹分析"],
        ["屏幕坐标", "screen_x_norm, screen_y_norm, hit_ui_plane", "2D界面坐标与有效命中判断"],
        ["AOI标签", "aoi_id, aoi_valid, aoi_name, target_controls", "控件0-9标签、多目标监督信号"],
        ["生理信号", "pupil_diameter, pupil_valid, openness", "注意力、眨眼、无效帧建模"],
    ])
    add_text(slide, "数据导出：Frame_*.csv 记录帧级数据；Summary_*.csv 记录轮次级注视统计；Eval*.csv 记录算法窗口评估结果。", 64, 382, 820, 52, size=12, color=INK)

    slide = pres.Slides.Add(7, 12)
    add_bg(slide, "06 Transformer输入维度与特征组织", "Model Input")
    add_metric(slide, 68, 92, 132, 92, str(stats["feature_dim"]), "总特征维度", TEAL)
    add_metric(slide, 220, 92, 132, 92, str(stats["window_frames"]), "窗口帧数", GOLD)
    add_metric(slide, 372, 92, 132, 92, str(stats["step_frames"]), "推理步长", TEAL)
    add_metric(slide, 524, 92, 132, 92, str(stats["aoi_count"]), "控件类别", GOLD)
    add_metric(slide, 676, 92, 132, 92, str(stats["threshold"]), "输出阈值", TEAL)
    add_table(slide, 62, 226, 838, 178, ["维度段", "字段范围", "说明"], [
        ["连续物理特征", "0-13", "归一化坐标、位移、速度、视线方向、睁眼状态、瞳孔"],
        ["有效性标记", "14-16", "pupil_valid, tracking_valid, hit_ui_plane"],
        ["AOI独热编码", "17-26", "控件0-9的当前帧命中语义"],
        ["窗口统计特征", "27-46", "瞳孔/速度分位数、有效率、AOI转移率、熵、dwell分布"],
    ])
    add_text(slide, f"当前 stats：stage={stats['stage']}，dataset={stats['dataset']}，active features={stats['active_feature_count']}，ONNX={stats['onnx_size_kb']}KB。", 64, 430, 820, 24, size=10, color=MUTED)

    slide = pres.Slides.Add(8, 12)
    add_bg(slide, "07 三类算法并行设计", "Algorithm")
    add_table(slide, 54, 88, 852, 250, ["算法", "输入", "输出", "状态"], [
        ["规则法", "AOI命中、连续注视、瞳孔门控", "Top3控件 / 命中计数", "已实现"],
        ["惩罚加权递归贝叶斯", "注视距离、速度、稳定性、瞳孔Z分数", "后验概率 / 置信阈值", "已实现"],
        ["TinyEyeTransformer", "150x47时序窗口", "10维sigmoid概率", "ONNX推理已集成"],
    ])
    add_card(slide, 84, 370, 230, 80, "实时性", "规则/贝叶斯在 Unity 内逐窗计算；Transformer 使用 Sentis 加载 ONNX。", TEAL)
    add_card(slide, 364, 370, 230, 80, "鲁棒性", "动态任务通过概率平滑与时序模型降低瞬时噪声影响。", GOLD)
    add_card(slide, 644, 370, 230, 80, "可评估", "窗口级导出 Top1、Hit@3、Subset、延迟等指标。", TEAL)

    slide = pres.Slides.Add(9, 12)
    add_bg(slide, "08 TinyEyeTransformer模型参数", "Transformer")
    add_card(slide, 56, 88, 260, 132, "模型结构", "Linear Projection + Positional Encoding + 2层 Transformer Encoder + Weighted GAP + Widget-Query Cross Attention。", TEAL)
    add_card(slide, 350, 88, 260, 132, "输出形式", "10个控件对应10维多标签概率，使用 sigmoid 与阈值判定，不再采用运行时训练模式。", GOLD)
    add_card(slide, 644, 88, 260, 132, "部署约束", "Unity加载 eye_transformer_finetuned.onnx，并严格校验 stats 的150x47、AOI 0-9与feature order。", TEAL)
    add_table(slide, 74, 260, 812, 138, ["参数", "当前值", "说明"], [
        ["输入张量", "[1, 150, 47]", "单批次、3秒窗口、47维特征"],
        ["标签空间", "0-9 / 10类", "对应10个界面控件"],
        ["推理步长", "15帧", "约300ms更新一次推荐"],
        ["最小有效帧", "20帧", "低于阈值不输出决策"],
        ["阈值", "0.30", "超过阈值进入预测集合"],
    ])

    slide = pres.Slides.Add(10, 12)
    add_bg(slide, "09 训练与部署流程", "Pipeline")
    add_flow(slide, 88, [
        ("GazeBaseVR", "原始CSV\n250Hz"),
        ("预训练", "任务表征\n通用眼动模式"),
        ("Unity数据", "Frame CSV\n0-9标签"),
        ("微调导出", "ONNX\nstats JSON"),
    ])
    add_flow(slide, 250, [
        ("Unity加载", "Sentis\nModelAsset"),
        ("参数校验", "150x47\nstage=finetune"),
        ("实时推理", "Top1/Top3\n多标签概率"),
        ("评估报告", "Eval CSV\nSummary"),
    ])
    add_text(slide, "训练脚本：Tools/train_eye_transformer_modified.py；部署模型：Assets/Models/eye_transformer_finetuned.onnx；运行配置：Assets/Resources/eye_transformer_stats.json。", 80, 430, 800, 32, size=10, color=MUTED)

    slide = pres.Slides.Add(11, 12)
    add_bg(slide, "10 指标体系与完成情况", "Metrics")
    add_table(slide, 54, 90, 852, 250, ["指标", "含义", "当前实现状态"], [
        ["Top1 Accuracy", "首位推荐是否命中目标控件", "评估器已导出"],
        ["Hit@3", "前三推荐是否包含目标控件", "评估器已导出"],
        ["Subset Accuracy", "多标签预测集合是否完全一致", "Transformer评估已支持"],
        ["Recall / Macro-F1", "多标签召回与类别均衡表现", "训练脚本已支持"],
        ["First Correct Latency", "首次正确推荐距任务开始的延迟", "评估器已导出"],
    ])
    add_card(slide, 84, 372, 230, 86, "静态指标目标", "三类算法正确率均不低于90%。", TEAL)
    add_card(slide, 364, 372, 230, 86, "动态指标目标", "至少一种核心算法正确率不低于85%。", GOLD)
    add_card(slide, 644, 372, 230, 86, "待补齐证据", "需用完整10人/100次/20小时数据集形成正式验收表。", TEAL)

    slide = pres.Slides.Add(12, 12)
    add_bg(slide, "11 项目进度总结", "Progress")
    add_table(slide, 54, 88, 852, 296, ["阶段", "计划内容", "当前结论"], [
        ["方案设计", "实验任务、算法、数据集方案", "已完成"],
        ["平台搭建", "PICO集成、眼动映射、任务界面、数据采集", "已完成"],
        ["中期验证", "规则法、概率法、首批数据采集", "已完成"],
        ["系统集成", "Transformer模型、三算法联调、评估报告", "进行中，训练脚本和ONNX推理链路已补齐"],
        ["验收交付", "完整数据集、源码、模型、说明书、总结报告", "需继续补齐数据与真机验收结果"],
    ])
    add_text(slide, "阶段判断：工程能力已经进入系统集成与验证阶段，下一步重点不再是原型功能，而是数据规模、模型复训、真机稳定性和交付文档收口。", 72, 424, 820, 36, size=12, bold=True, color=INK)

    slide = pres.Slides.Add(13, 12)
    add_bg(slide, "12 下一步工作清单", "Next")
    add_card(slide, 62, 92, 250, 126, "数据集补齐", "完成不少于10名被试、100次实验、20小时有效数据，输出数据质检和划分清单。", TEAL)
    add_card(slide, 356, 92, 250, 126, "模型复训", "使用GazeBaseVR预训练权重和正式Unity数据微调，固化ONNX、stats、指标CSV。", GOLD)
    add_card(slide, 650, 92, 250, 126, "真机验收", "PICO端验证眼动权限、采样频率、推荐更新、确认交互与数据导出路径。", TEAL)
    add_table(slide, 92, 266, 776, 120, ["交付物", "形式"], [
        ["完善版汇报PPT", "项目现状、参数维度、指标体系、进度总结"],
        ["软件说明书", "架构、需求、数据结构、使用方式、PICO操作说明"],
        ["算法对比报告", "规则/贝叶斯/Transformer静态与动态任务指标"],
    ])

    pres.SaveAs(str(PPT_OUT))
    pres.Close()
    ppt.Quit()


def add_word_heading(doc, text, level=1):
    p = doc.Paragraphs.Add()
    p.Range.Text = text
    p.Range.Font.Name = "Microsoft YaHei"
    p.Range.Font.Size = 16 if level == 1 else 13
    p.Range.Font.Bold = -1
    p.Range.Font.Color = NAVY if level == 1 else TEAL
    p.Range.InsertParagraphAfter()


def add_word_para(doc, text, bold=False):
    p = doc.Paragraphs.Add()
    p.Range.Text = text
    p.Range.Font.Name = "Microsoft YaHei"
    p.Range.Font.Size = 10.5
    p.Range.Font.Bold = -1 if bold else 0
    p.Range.Font.Color = INK
    p.Range.InsertParagraphAfter()


def add_word_bullets(doc, items):
    for item in items:
        p = doc.Paragraphs.Add()
        p.Range.Text = "• " + item
        p.Range.Font.Name = "Microsoft YaHei"
        p.Range.Font.Size = 10.5
        p.Range.Font.Color = INK
        p.Range.InsertParagraphAfter()


def add_word_table(doc, headers, rows):
    rng = doc.Range(doc.Content.End - 1, doc.Content.End - 1)
    table = doc.Tables.Add(rng, len(rows) + 1, len(headers))
    table.Borders.Enable = True
    for c, h in enumerate(headers, 1):
        cell = table.Cell(1, c)
        cell.Range.Text = h
        cell.Range.Font.Bold = -1
        cell.Range.Font.Name = "Microsoft YaHei"
        cell.Range.Font.Color = WHITE
        cell.Shading.BackgroundPatternColor = NAVY
    for r, row in enumerate(rows, 2):
        for c, value in enumerate(row, 1):
            cell = table.Cell(r, c)
            cell.Range.Text = str(value)
            cell.Range.Font.Name = "Microsoft YaHei"
            cell.Range.Font.Size = 9
            cell.Shading.BackgroundPatternColor = WHITE
    doc.Paragraphs.Add().Range.InsertParagraphAfter()


def build_doc(stats: dict):
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Add()
    doc.PageSetup.TopMargin = 54
    doc.PageSetup.BottomMargin = 54
    doc.PageSetup.LeftMargin = 54
    doc.PageSetup.RightMargin = 54

    add_word_heading(doc, "交互组合仿真子系统软件说明书", 1)
    add_word_para(doc, "版本：V1.0    日期：2026年5月10日    适用设备：PICO 4 Pro / Windows主机 / Unity 2022.3", True)
    add_word_para(doc, "本文档用于说明基于眼动追踪的意图识别实验软件的架构、需求、数据结构、操作流程、PICO设备使用方式和模型部署方法。")

    add_word_heading(doc, "1. 软件概述", 1)
    add_word_para(doc, "本软件面向VR/AR环境下的实时眼动意图识别任务，利用PICO 4 Pro采集眼动信号，在Unity中完成任务派发、AOI命中、数据记录、算法推理和结果评估。系统采用规则推理、惩罚加权递归贝叶斯和TinyEyeTransformer三类算法并行验证。")

    add_word_heading(doc, "2. 需求分析", 1)
    add_word_bullets(doc, [
        "功能需求：支持静态任务、动态任务、10个AOI控件、手柄确认、眼动数据采集、三类算法推理与评估导出。",
        "性能需求：静态任务三类算法正确率目标不低于90%；动态任务至少一种核心算法正确率目标不低于85%。",
        "数据需求：最终数据集不少于10名被试、100次实验、20小时有效眼动数据。",
        "集成需求：PICO设备与Windows主机稳定连接，眼动坐标可映射至Unity世界空间UI平面。",
    ])

    add_word_heading(doc, "3. 软件架构", 1)
    add_word_table(doc, ["层级", "模块", "说明"], [
        ["设备层", "PICO 4 Pro", "采集眼动、头显姿态、手柄输入"],
        ["接入层", "PXR SDK / XR Interaction Toolkit", "眼动权限、设备状态、XR交互事件"],
        ["任务层", "TaskManager / ControlItem", "任务生成、控件状态、确认流程、动态干扰"],
        ["采集层", "EyeTrackingManager / GazeDataLogger", "坐标映射、AOI命中、CSV导出"],
        ["算法层", "RealtimeIntentionRecommender / OnnxIntentionInference", "规则法、贝叶斯法、ONNX推理"],
        ["评估层", "WindowPredictionEvaluator", "窗口级准确率、Hit@3、Subset、延迟导出"],
    ])

    add_word_heading(doc, "4. 运行环境与部署文件", 1)
    add_word_bullets(doc, [
        "Unity版本：Unity 2022.3。",
        "XR依赖：PICO Unity Integration SDK、XR Interaction Toolkit、Unity Sentis。",
        "核心场景：Assets/Scenes/DemoScene.unity。",
        "模型文件：Assets/Models/eye_transformer_finetuned.onnx。",
        "模型配置：Assets/Resources/eye_transformer_stats.json。",
        "训练脚本：Tools/train_eye_transformer_modified.py。",
        "数据同步脚本：Tools/sync_eye_csv_from_headset.ps1。",
    ])

    add_word_heading(doc, "5. 数据结构", 1)
    add_word_table(doc, ["类别", "字段", "说明"], [
        ["会话信息", "subject_id, session_id, round_index, mode, phase", "标识被试、实验轮次和任务阶段"],
        ["时间信息", "timestamp_ms, frame_idx", "用于窗口切分和延迟计算"],
        ["三维视线", "gaze_origin_*, gaze_dir_*, gaze_point_*", "描述眼动射线在空间中的位置和方向"],
        ["二维映射", "screen_x_norm, screen_y_norm, hit_ui_plane", "归一化屏幕坐标和UI平面命中状态"],
        ["AOI标签", "aoi_id, aoi_valid, aoi_name, target_controls", "控件0-9、有效命中、多目标监督标签"],
        ["生理指标", "pupil_diameter, pupil_valid, left_openness, right_openness", "瞳孔、睁眼状态和无效帧判定"],
    ])

    add_word_heading(doc, "6. Transformer模型说明", 1)
    add_word_table(doc, ["项目", "当前配置"], [
        ["训练方式", "GazeBaseVR预训练 + Unity Frame CSV微调"],
        ["输入张量", "[1, 150, 47]"],
        ["窗口长度", f"{stats['window_frames']}帧，约3秒"],
        ["推理步长", f"{stats['step_frames']}帧"],
        ["最小有效帧", f"{stats['min_valid_frames']}帧"],
        ["标签空间", "控件0-9，共10类"],
        ["输出形式", "10维sigmoid多标签概率"],
        ["阈值", str(stats["threshold"])],
        ["ONNX大小", f"{stats['onnx_size_kb']} KB"],
    ])
    add_word_para(doc, "Unity端不进行运行时训练。OnnxIntentionInference会检查stats文件的stage、feature_dim、window_frames、aoi_label_base和特征顺序，防止模型与运行时输入不一致。")

    add_word_heading(doc, "7. PICO设备操作说明", 1)
    add_word_bullets(doc, [
        "开机并佩戴PICO 4 Pro，调整头带和瞳距，确保画面清晰且视线追踪稳定。",
        "进入系统设置或应用授权流程，允许眼动追踪相关权限。应用侧会尝试请求com.picovr.permission.EYE_TRACKING。",
        "实验前进行眼动校准，校准未通过时不进入正式采集。",
        "连接Windows主机进行开发或日志拉取时，确认USB调试/ADB连接可用。",
        "实验过程中保持头部自然稳定，允许轻微移动；任务完成由手柄确认，不依赖注视直接触发。",
    ])

    add_word_heading(doc, "8. 软件使用流程", 1)
    add_word_bullets(doc, [
        "在Unity中打开DemoScene，确认EyeTrackingManager、GazeDataLogger、TaskManager、OnnxIntentionInference引用完整。",
        "设置subjectId和sessionId，选择静态或动态模式，选择实验轮次。",
        "点击Start Experiment开始实验。动态模式会先进入自由浏览阶段，然后随机释放任务。",
        "被试看向目标控件，控件值达到阈值且符号为1后，使用手柄或确认按钮完成当前轮次。",
        "实验结束后系统导出Frame、Summary和Eval相关CSV文件。",
    ])

    add_word_heading(doc, "9. 数据导出与模型训练", 1)
    add_word_para(doc, "GazeDataLogger默认在F:\\TestData或Android应用数据目录导出CSV。PICO端数据可通过Tools/sync_eye_csv_from_headset.ps1同步至主机。")
    add_word_bullets(doc, [
        "GazeBaseVR预训练命令：python Tools\\train_eye_transformer_modified.py --stage pretrain --gazebasevr_dir <数据目录> --out_dir F:\\TestData\\OUT_DIR",
        "Unity微调命令：python Tools\\train_eye_transformer_modified.py --stage finetune --frame_dir F:\\TestData --pretrained_ckpt F:\\TestData\\OUT_DIR\\eye_transformer_pretrained.pt --export_onnx Assets\\Models\\eye_transformer_finetuned.onnx --stats_json Assets\\Resources\\eye_transformer_stats.json --aoi_label_base 0",
        "训练后需将ONNX和stats文件一起放入Unity工程，二者必须匹配。",
    ])

    add_word_heading(doc, "10. 指标评估", 1)
    add_word_table(doc, ["指标", "说明"], [
        ["Top1 Accuracy", "首位推荐控件是否命中任务目标"],
        ["Hit@3", "前三推荐是否包含任一任务目标"],
        ["Subset Accuracy", "预测集合与多目标标签是否完全一致"],
        ["Macro-F1 / Recall", "多标签分类质量和召回能力"],
        ["First Correct Latency", "首次正确推荐距离任务开始的延迟"],
    ])

    add_word_heading(doc, "11. 常见问题处理", 1)
    add_word_bullets(doc, [
        "无眼动数据：检查PICO权限、设备是否支持眼动、PXR SDK是否正常初始化。",
        "AOI无法命中：检查uiPlaneRect、controlItems引用和控件编号0-9是否一致。",
        "Transformer不输出：检查stats是否为finetune、feature_dim是否为47、ONNX输入是否为150x47。",
        "CSV为空或缺字段：检查GazeDataLogger是否挂载，输出目录是否有写入权限。",
        "动态任务难以复现：启用TaskManager中的useDeterministicSimulationSeed并记录simulationSeed。",
    ])

    add_word_heading(doc, "12. 当前进度总结", 1)
    add_word_para(doc, "截至2026年5月10日，系统已具备完整实验平台、眼动采集、任务管理、规则算法、递归贝叶斯、Transformer离线训练脚本、ONNX推理接入、运行时UI美化和窗口级评估导出能力。后续重点是补齐正式数据集规模、完成模型复训、进行PICO真机验收并形成算法对比报告。")

    doc.SaveAs(str(DOC_OUT))
    doc.Close()
    word.Quit()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = read_stats()
    build_ppt(stats)
    build_doc(stats)
    print(PPT_OUT)
    print(DOC_OUT)


if __name__ == "__main__":
    main()
