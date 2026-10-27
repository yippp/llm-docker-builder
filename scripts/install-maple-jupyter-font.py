#!/usr/bin/env python3
"""装 Maple Mono NL CN，并把它设成 JupyterLab 的代码/终端字体。"""

import json
import os
import shutil
import subprocess
from pathlib import Path

SRC = Path("/tmp/maple")
FONT_DIR = Path("/usr/share/fonts/truetype/maple")
WEB_DIR_NAME = "fonts"
MARKER = "/* maple-mono-nl-cn */"
KEEP_SUFFIX = ("-Regular.ttf", "-Bold.ttf", "-Italic.ttf", "-BoldItalic.ttf")


def keep(name: str) -> bool:
    return name.endswith(KEEP_SUFFIX) and "Variable" not in name


def family_of(path: Path) -> str:
    out = subprocess.check_output(
        ["fc-scan", "--format", "%{family}\n", str(path)], text=True
    )
    names = [part.strip() for part in out.split(",") if part.strip()]
    for name in names:
        if "Maple" in name and "CN" in name:
            return name
    if not names:
        raise SystemExit(f"读不到字体家族名: {path}")
    return names[0]


def face(family: str, filename: str, weight: int, style: str) -> str:
    # 主题 CSS 在 .../lab/api/themes/<name>/index.css，相对路径回到 /static/lab/
    url = f"../../../../static/lab/{WEB_DIR_NAME}/{filename}"
    return (
        "@font-face {\n"
        f"  font-family: {json.dumps(family)};\n"
        f"  src: url({json.dumps(url)}) format(\"woff2\");\n"
        f"  font-weight: {weight};\n"
        f"  font-style: {style};\n"
        "  font-display: swap;\n"
        "}\n"
    )


def main() -> None:
    ttfs = [p for p in SRC.rglob("*.ttf") if keep(p.name)]
    if not any(p.name.endswith("-Regular.ttf") for p in ttfs):
        names = sorted(p.name for p in SRC.rglob("*.ttf"))
        raise SystemExit("压缩包里没有 Regular：" + ", ".join(names[:30]))

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for src in ttfs:
        dst = FONT_DIR / src.name
        shutil.copy2(src, dst)
        subprocess.check_call(["woff2_compress", str(dst)])

    # woff2_compress 生成 name.ttf.woff2
    web_fonts = sorted(FONT_DIR.glob("*.woff2"))
    if not web_fonts:
        raise SystemExit("woff2 没生成")

    subprocess.check_call(["fc-cache", "-f", str(FONT_DIR)])
    regular = next(p for p in FONT_DIR.glob("*-Regular.ttf"))
    family = family_of(regular)
    print("字体家族:", family)

    app = subprocess.check_output(["jupyter", "lab", "path"], text=True)
    app_dir = ""
    for line in app.splitlines():
        if line.startswith("Application directory:"):
            app_dir = line.split(":", 1)[1].strip()
    if not app_dir:
        raise SystemExit("jupyter lab path 没有 Application directory")
    app_path = Path(app_dir)
    web_dir = app_path / "static" / WEB_DIR_NAME
    web_dir.mkdir(parents=True, exist_ok=True)
    for font in web_fonts:
        shutil.copy2(font, web_dir / font.name)

    css = MARKER + "\n"
    for font in web_fonts:
        stem = font.name
        for suffix in (".ttf.woff2", ".woff2"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem.endswith("-Regular"):
            css += face(family, font.name, 400, "normal")
        elif stem.endswith("-Bold"):
            css += face(family, font.name, 700, "normal")
        elif stem.endswith("-BoldItalic"):
            css += face(family, font.name, 700, "italic")
        elif stem.endswith("-Italic"):
            css += face(family, font.name, 400, "italic")
    themes = list((app_path / "themes").glob("*/index.css"))
    if not themes:
        raise SystemExit(f"没找到主题 CSS: {app_path / 'themes'}")
    for theme in themes:
        text = theme.read_text(encoding="utf-8", errors="replace")
        if MARKER not in text:
            theme.write_text(text.rstrip() + "\n\n" + css, encoding="utf-8")

    stack = f"{family}, monospace"
    css_stack = f'"{family}", monospace'
    overrides = {
        "@jupyterlab/apputils-extension:themes": {
            "theme": "JupyterLab Dark",
            "overrides": {"code-font-family": css_stack},
        },
        "@jupyterlab/terminal-extension:plugin": {
            "fontFamily": stack,
            "theme": "dark",
        },
        "@jupyterlab/codemirror-extension:plugin": {
            "defaultConfig": {"fontFamily": stack},
        },
    }
    payload = json.dumps(overrides, ensure_ascii=False, indent=2) + "\n"
    settings = app_path / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    (settings / "overrides.json").write_text(payload, encoding="utf-8")
    Path("/opt/jupyter").mkdir(parents=True, exist_ok=True)
    Path("/opt/jupyter/lab-overrides.json").write_text(payload, encoding="utf-8")
    print("已写入", settings / "overrides.json")


if __name__ == "__main__":
    main()
