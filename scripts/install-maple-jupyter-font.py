#!/usr/bin/env python3
"""装 Maple Mono NL CN，并把它设成 JupyterLab 的代码/终端字体。"""

import json
import shutil
import subprocess
from pathlib import Path

SRC = Path("/tmp/maple")
FONT_DIR = Path("/usr/share/fonts/truetype/maple")
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
    # custom.css 由 /custom/custom.css 提供，字体放在同一目录，用相对路径。
    return (
        "@font-face {\n"
        f"  font-family: {json.dumps(family)};\n"
        f"  src: url({json.dumps(filename)}) format(\"woff2\");\n"
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
    if "@font-face" not in css:
        raise SystemExit("没有可嵌入的 woff2")

    # pip 装的 JupyterLab 没有 share/jupyter/lab/themes。字体放到 jupyter_server
    # 默认会搜索的 static/custom，由 /custom/custom.css 提供，不依赖 HOME。
    import jupyter_server

    custom_dirs = [
        Path(jupyter_server.__file__).resolve().parent / "static" / "custom",
        Path("/opt/jupyter/custom"),
    ]
    for dest in custom_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "custom.css").write_text(css, encoding="utf-8")
        for font in web_fonts:
            shutil.copy2(font, dest / font.name)
    print("字体 CSS:", custom_dirs[0] / "custom.css")

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
    app = subprocess.check_output(["jupyter", "lab", "path"], text=True)
    app_dir = ""
    for line in app.splitlines():
        if line.startswith("Application directory:"):
            app_dir = line.split(":", 1)[1].strip()
    if not app_dir:
        raise SystemExit("jupyter lab path 没有 Application directory")
    settings = Path(app_dir) / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    (settings / "overrides.json").write_text(payload, encoding="utf-8")
    Path("/opt/jupyter/lab-overrides.json").write_text(payload, encoding="utf-8")
    print("已写入", settings / "overrides.json")


if __name__ == "__main__":
    main()
