#!/usr/bin/env python3
"""Render Markdown notes as print-ready A4 PDFs.

Usage:
    python md_to_pdf.py [FILE.md ...]

When no files are supplied, every Markdown file in this directory is rendered.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent
FONT = Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf")
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

TITLES = {
    "triple_barrier_impl_discussion": "Triple Barrier Implementation Discussion",
    "volatility-dynamic-barriers": "Volatility and Dynamic Barriers",
    "why_need_dynamic_thresholds": "왜 동적 임계값이 필요한가?",
}


def _math_to_text(text: str) -> str:
    """Keep the small amount of TeX in these notes readable without JavaScript."""
    substitutions = {
        r"\\times": "×",
        r"\times": "×",
        r"\\sigma": "σ",
        r"\sigma": "σ",
        r"\\tau": "τ",
        r"\tau": "τ",
    }
    for source, target in substitutions.items():
        text = text.replace(source, target)
    text = re.sub(r"\\{1,2}\((.*?)\\{1,2}\)", r"<span class=\"math\">\1</span>", text)
    text = re.sub(r"\$([^$\n]+)\$", r"<span class=\"math\">\1</span>", text)
    return text


def _format_pipeline_output(text: str) -> str:
    marker = "================================================================================"
    start = text.find(marker)
    if start < 0:
        return text
    end_marker = "이것이 마르코스 데 프라도 교수가 제시한 '정밀도와 재현율의 트레이드오프' 원리입니다."
    end = text.find(end_marker, start)
    if end < 0:
        return text
    end += len(end_marker)
    block = text[start:end]
    return text[:start].rstrip() + "\n\n```text\n" + block.strip() + "\n```\n\n" + text[end:].lstrip()


def _repair_volatility_note(text: str) -> str:
    first_line, _, body = text.lstrip().partition("\n")
    missing_notice = ""
    if first_line.startswith("[volatility barriers]"):
        missing = first_line[first_line.find("(") + 1 : first_line.rfind(")")]
        missing_notice = (
            f'<div class="missing-image">도표 파일 없음: '
            f'<code>{html.escape(missing)}</code></div>'
        )
        text = body.lstrip()

    text = text.replace(
        "🚀 시뮬레이션 및 ML 모델 구동 검증 결과실제 데이터를 통해",
        "# 시뮬레이션 및 ML 모델 구동 검증 결과\n\n실제 데이터를 통해",
    )
    if missing_notice:
        title_line = "# 시뮬레이션 및 ML 모델 구동 검증 결과"
        text = text.replace(title_line, title_line + "\n\n" + missing_notice, 1)
    text = text.replace(
        "이뤄냅니다.📂 완성된 실습 파이프라인(dollar-bar-triple-barrier-pipeline.py) 코드 구조스크립트는",
        "이뤄냅니다.\n\n## 완성된 실습 파이프라인 구조\n\n"
        "`dollar-bar-triple-barrier-pipeline.py` 스크립트는",
    )
    for label in ("1차 모델 (Primary Model)", "2차 모델 (Secondary / Meta-Labeling Model)", "결과적 강점"):
        text = text.replace(label + ":", "\n\n## " + label + "\n\n")
    text = re.sub(r"(?<!# )Step ([1-6]):\s*", r"\n\n### Step \1\n\n", text)
    text = re.sub(r"Step 7~8:\s*", r"\n\n### Step 7~8\n\n", text)

    marker = "================================================================================"
    start = text.find(marker)
    if start >= 0:
        third = start
        for _ in range(3):
            third = text.find(marker, third + (1 if third == start else len(marker)))
            if third < 0:
                break
        # There are three separator rows total; locate the last one directly.
        rows = [m.start() for m in re.finditer(re.escape(marker), text[start:])]
        if len(rows) >= 3:
            end = start + rows[2] + len(marker)
            block = text[start:end]
            text = text[:start].rstrip() + "\n\n```text\n" + block + "\n```\n\n" + text[end:].lstrip()
    return text


def prepare_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.stem == "volatility-dynamic-barriers":
        text = _repair_volatility_note(text)
    elif path.stem == "triple_barrier_impl_discussion":
        text = text.replace("# what is going on?", "# What Is Going On?", 1)
        text = _format_pipeline_output(text)

    if not re.search(r"^#\s+", text, re.MULTILINE):
        text = f"# {TITLES.get(path.stem, path.stem.replace('_', ' ').title())}\n\n{text.lstrip()}"
    return _math_to_text(text)


def render(path: Path) -> Path:
    title = TITLES.get(path.stem, path.stem.replace("_", " ").title())
    body = markdown.markdown(
        prepare_markdown(path),
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )
    font_rule = ""
    if FONT.exists():
        font_rule = f"@font-face {{ font-family: 'PDF Gothic'; src: url('{FONT.as_uri()}'); }}"
    css = f"""
        {font_rule}
        @page {{
          size: A4;
          margin: 19mm 18mm 20mm;
          background: white;
          @bottom-left {{ content: "{title}"; font-size: 8pt; color: #64748b; }}
          @bottom-right {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #64748b; }}
        }}
        html {{ font-family: 'PDF Gothic', sans-serif; color: #172033; font-size: 10.5pt; background: white; }}
        body {{ line-height: 1.62; overflow-wrap: anywhere; background: white; }}
        h1 {{ font-size: 23pt; line-height: 1.28; color: #123a63; margin: 0 0 9mm; padding-bottom: 4mm; border-bottom: 2px solid #2d6da3; }}
        h2 {{ font-size: 15pt; line-height: 1.35; color: #184e77; margin: 8mm 0 3mm; break-after: avoid; }}
        h3 {{ font-size: 12pt; line-height: 1.4; color: #245f8f; margin: 6mm 0 2mm; break-after: avoid; }}
        p {{ margin: 0 0 3.2mm; orphans: 2; widows: 2; }}
        ul, ol {{ margin: 1.5mm 0 4mm; padding-left: 7mm; }}
        li {{ margin: 1.2mm 0; }}
        strong {{ color: #0f3555; }}
        code {{ font-family: 'D2Coding', 'Courier New', monospace; font-size: 0.9em; background: #eef3f7; padding: 0.2em 0.35em; border-radius: 3px; }}
        pre {{ white-space: pre-wrap; font-family: 'D2Coding', 'Courier New', monospace; font-size: 8.2pt; line-height: 1.48; background: #f4f7fa; border: 1px solid #d8e1e8; border-left: 4px solid #2d6da3; padding: 4mm; break-inside: avoid; }}
        pre code {{ background: none; padding: 0; }}
        .math {{ font-family: 'Times New Roman', 'PDF Gothic', serif; font-style: italic; white-space: nowrap; }}
        .missing-image {{ border: 1px dashed #b7c2cc; background: #f8fafc; color: #64748b; padding: 4mm; margin-bottom: 6mm; text-align: center; }}
        hr {{ border: 0; border-top: 1px solid #cbd5e1; margin: 7mm 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 4mm 0; font-size: 9.5pt; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 2mm; vertical-align: top; }}
        th {{ background: #eaf2f8; color: #123a63; }}
        img {{ display: block; max-width: 100%; max-height: 225mm; margin: 5mm auto; }}
    """
    document = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
    <title>{html.escape(title)}</title><style>{css}</style></head><body>{body}</body></html>"""
    output = path.with_suffix(".pdf")
    if not CHROME.exists():
        raise RuntimeError(f"PDF renderer not found: {CHROME}")
    with tempfile.TemporaryDirectory(prefix="md-to-pdf-") as temp_dir:
        temp = Path(temp_dir)
        html_path = temp / "document.html"
        html_path.write_text(document, encoding="utf-8")
        subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                f"--print-to-pdf={output}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    files = args.files or sorted(ROOT.glob("*.md"))
    for item in files:
        source = item if item.is_absolute() else ROOT / item
        if source.name == Path(__file__).name or source.suffix.lower() != ".md":
            continue
        print(render(source.resolve()))


if __name__ == "__main__":
    main()
