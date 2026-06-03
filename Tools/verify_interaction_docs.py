from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:\UnityProgram\test2")
OUT_DIR = ROOT / "deliverables" / "docx_draft"
RENDER_DIR = ROOT / "deliverables" / "docx_render_check"
RENDERER = Path(
    r"C:\Users\Lenovo\.codex\plugins\cache\openai-primary-runtime\documents\26.515.10909\skills\documents\render_docx.py"
)
PYTHON = Path(r"C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
SYSTEM_NAME = "交互组合仿真子系统"
EXPECTED = [
    "交互组合仿真子系统需求分析报告.docx",
    "交互组合仿真子系统软件设计方案.docx",
    "交互组合仿真子系统设计方案报告.docx",
    "交互组合仿真子系统程序员手册.docx",
    "交互组合仿真子系统用户手册.docx",
    "交互组合仿真子系统操作规程.docx",
    "交互组合仿真子系统使用维护说明书.docx",
    "交互组合仿真子系统项目测试大纲.docx",
    "交互组合仿真子系统项目测试说明.docx",
    "交互组合仿真子系统测试报告.docx",
    "交互组合仿真子系统调试报告.docx",
    "交互组合仿真子系统项目研制总结.docx",
]


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts: list[str] = []
    for para in root.findall(".//w:p", ns):
        text = "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def make_contact_sheet(pages: list[Path], out_path: Path, title: str):
    if not pages:
        return
    thumbs: list[Image.Image] = []
    for page in pages:
        img = Image.open(page).convert("RGB")
        img.thumbnail((420, 594), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (460, 650), "white")
        canvas.paste(img, ((460 - img.width) // 2, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 8), page.name, fill=(30, 30, 30))
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows * 650 + 40), (242, 244, 247))
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 12), title, fill=(20, 40, 70))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 460
        y = 40 + (idx // cols) * 650
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    structural_only = "--structural-only" in sys.argv
    report: dict = {"structural": [], "render": []}
    ok = True
    for name in EXPECTED:
        path = OUT_DIR / name
        item = {"name": name, "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
        if path.exists():
            text = docx_text(path)
            item["contains_system_name"] = SYSTEM_NAME in text
            item["contains_title_keyword"] = name.removesuffix(".docx").replace(SYSTEM_NAME, "") in text
            item["paragraph_like_text_length"] = len(text)
            if not item["contains_system_name"] or item["paragraph_like_text_length"] < 1200:
                ok = False
        else:
            ok = False
        report["structural"].append(item)

    if structural_only:
        report["render_skipped"] = "structural-only mode"
    elif RENDERER.exists():
        for name in EXPECTED:
            path = OUT_DIR / name
            target = RENDER_DIR / path.stem
            target.mkdir(parents=True, exist_ok=True)
            cmd = [str(PYTHON), str(RENDERER), str(path), "--output_dir", str(target)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            pages = sorted(target.glob("page-*.png"))
            contact = target / "contact_sheet.png"
            make_contact_sheet(pages, contact, path.stem)
            record = {
                "name": name,
                "returncode": proc.returncode,
                "page_png_count": len(pages),
                "contact_sheet": str(contact) if contact.exists() else "",
                "stdout_tail": proc.stdout[-1000:],
                "stderr_tail": proc.stderr[-1000:],
            }
            if proc.returncode != 0 or not pages:
                ok = False
            report["render"].append(record)
    else:
        ok = False
        report["render_error"] = f"renderer not found: {RENDERER}"

    report_path = OUT_DIR / "verification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
