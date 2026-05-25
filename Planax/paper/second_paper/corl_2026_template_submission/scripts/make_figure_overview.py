#!/usr/bin/env python3
"""Generate the vector overview figure for the CoRL paper.

The source PNG is copied only as a draft/reference artifact. The final figure is
pure SVG vector graphics and contains no raster <image> elements.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import svgwrite


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SRC_DRAFT = ROOT / "图片" / "figure1.png"
DRAFT_COPY = FIG_DIR / "overview_draft.png"
SVG_OUT = FIG_DIR / "figure_overview.svg"
PDF_OUT = FIG_DIR / "figure_overview.pdf"
PNG_OUT = FIG_DIR / "figure_overview.png"

W, H = 1600, 900
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"

BLUE = "#1f5fbf"
ORANGE = "#f26b21"
DARK = "#444444"
GREEN = "#22864b"
LINE = "#9a9a9a"
LIGHT_LINE = "#c9c9c9"
LIGHT_BG = "#f7f7f7"
TEXT = "#222222"
MUTED = "#707070"
PANEL_FILL = "#ffffff"


def rotate_point(px: float, py: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return (
        px * math.cos(a) - py * math.sin(a),
        px * math.sin(a) + py * math.cos(a),
    )


def arrow(dwg: svgwrite.Drawing, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 6) -> None:
    """Draw a clean line arrow with a filled triangular head."""
    dwg.add(dwg.line((x1, y1), (x2, y2), stroke=color, stroke_width=width, stroke_linecap="round"))
    ang = math.atan2(y2 - y1, x2 - x1)
    head_len = 18
    head_w = 18
    tip = (x2, y2)
    base = (x2 - head_len * math.cos(ang), y2 - head_len * math.sin(ang))
    left = (
        base[0] + (head_w / 2) * math.sin(ang),
        base[1] - (head_w / 2) * math.cos(ang),
    )
    right = (
        base[0] - (head_w / 2) * math.sin(ang),
        base[1] + (head_w / 2) * math.cos(ang),
    )
    dwg.add(dwg.polygon([tip, left, right], fill=color))


def text(
    dwg: svgwrite.Drawing,
    s: str,
    x: float,
    y: float,
    size: int = 18,
    fill: str = TEXT,
    weight: str | None = None,
    anchor: str = "start",
) -> None:
    kwargs = {
        "insert": (x, y),
        "font_family": FONT,
        "font_size": size,
        "fill": fill,
        "text_anchor": anchor,
    }
    if weight:
        kwargs["font_weight"] = weight
    dwg.add(dwg.text(s, **kwargs))


def rounded_rect(
    dwg: svgwrite.Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    stroke: str = LIGHT_LINE,
    fill: str = "#ffffff",
    sw: float = 1.4,
    rx: float = 8,
) -> None:
    dwg.add(dwg.rect(insert=(x, y), size=(w, h), rx=rx, ry=rx, fill=fill, stroke=stroke, stroke_width=sw))


def add_panel(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float, title: str, color: str) -> None:
    rounded_rect(dwg, x, y, w, h, stroke=color, fill=PANEL_FILL, sw=2.2, rx=11)
    title_size = 23 if len(title) > 17 else 26
    text(dwg, title, x + w / 2, y + 45, size=title_size, fill=color, weight="700", anchor="middle")


def add_plot_box(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    rounded_rect(dwg, x, y, w, h, stroke=LIGHT_LINE, fill="#ffffff", sw=1.3, rx=7)


def add_info_box(
    dwg: svgwrite.Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[str],
    bullet: str | None = "square",
    color: str = MUTED,
    size: int = 15,
) -> None:
    rounded_rect(dwg, x, y, w, h, stroke=LIGHT_LINE, fill="#ffffff", sw=1.2, rx=7)
    if not lines:
        return
    n = len(lines)
    gap = min(22, max(17, (h - 18) / max(n, 1)))
    y0 = y + 24 if n <= 3 else y + 22
    for i, line in enumerate(lines):
        yy = y0 + i * gap
        if bullet == "square":
            dwg.add(dwg.rect(insert=(x + 17, yy - 10), size=(9, 9), rx=1.4, fill=LINE))
            tx = x + 36
        elif bullet == "circle":
            dwg.add(dwg.circle(center=(x + 22, yy - 5), r=5.2, fill=LINE))
            tx = x + 38
        else:
            tx = x + 18
        text(dwg, line, tx, yy, size=size, fill=color)


def add_placeholder_lines(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    rounded_rect(dwg, x, y, w, h, stroke=LIGHT_LINE, fill="#ffffff", sw=1.2, rx=7)
    rounded_rect(dwg, x + 18, y + 18, 56, h - 36, stroke=LIGHT_LINE, fill=LIGHT_BG, sw=1.1, rx=5)
    for i, length in enumerate([42, 36, 29]):
        dwg.add(dwg.line((x + 31, y + 31 + 10 * i), (x + 31 + length, y + 31 + 10 * i), stroke=LIGHT_LINE, stroke_width=2))
    for i, length in enumerate([140, 116, 82]):
        yy = y + 29 + i * 18
        dwg.add(dwg.line((x + 95, yy), (x + 95 + length, yy), stroke=LINE, stroke_width=5, stroke_linecap="round", opacity=0.7))


def add_airplane_icon(dwg: svgwrite.Drawing, cx: float, cy: float, scale: float, angle: float, color: str = "#777777") -> None:
    g = dwg.g(transform=f"translate({cx},{cy}) rotate({angle}) scale({scale})")
    g.add(dwg.path(d="M 19 0 L -12 -7 L -4 0 L -12 7 Z", fill=color, stroke=color, stroke_width=1.2, stroke_linejoin="round"))
    g.add(dwg.path(d="M -8 0 L -17 -11 M -8 0 L -17 11", fill="none", stroke=color, stroke_width=3, stroke_linecap="round"))
    dwg.add(g)


def draw_reference(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    px, py = x + 18, y + 83
    pw, ph = w - 36, 224
    add_plot_box(dwg, px, py, pw, ph)
    stroke = "#777777"
    dash = "10,9"
    dwg.add(dwg.path(d=f"M {px+22} {py+78} C {px+84} {py+42}, {px+159} {py+29}, {px+247} {py+72}", fill="none", stroke=stroke, stroke_width=2.2, stroke_dasharray=dash))
    dwg.add(dwg.path(d=f"M {px+24} {py+152} C {px+105} {py+154}, {px+101} {py+64}, {px+156} {py+114} C {px+206} {py+160}, {px+199} {py+72}, {px+254} {py+111}", fill="none", stroke=stroke, stroke_width=2.2, stroke_dasharray=dash))
    dwg.add(dwg.path(d=f"M {px+22} {py+198} C {px+97} {py+186}, {px+116} {py+224}, {px+188} {py+205} C {px+222} {py+195}, {px+244} {py+213}, {px+267} {py+210}", fill="none", stroke=stroke, stroke_width=2.2, stroke_dasharray=dash))
    add_airplane_icon(dwg, px + 263, py + 61, 0.82, -28)
    add_airplane_icon(dwg, px + 255, py + 118, 0.72, -45)
    add_airplane_icon(dwg, px + 247, py + 201, 0.80, 28)
    add_airplane_icon(dwg, px + 36, py + 186, 0.70, -5)
    add_placeholder_lines(dwg, x + 18, y + 330, w - 36, 88)
    add_placeholder_lines(dwg, x + 18, y + 438, w - 36, 88)


def add_arc_arrow(dwg: svgwrite.Drawing, d: str, head: tuple[float, float, float], color: str, width: float = 4) -> None:
    dwg.add(dwg.path(d=d, fill="none", stroke=color, stroke_width=width, stroke_linecap="round"))
    cx, cy, angle = head
    g = dwg.g(transform=f"translate({cx},{cy}) rotate({angle})")
    g.add(dwg.polygon([(0, 0), (-16, -8), (-12, 0), (-16, 8)], fill=color))
    dwg.add(g)


def draw_rhtso(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    px, py = x + 18, y + 83
    pw, ph = w - 36, 224
    add_plot_box(dwg, px, py, pw, ph)
    cx, cy = px + pw / 2 - 10, py + ph / 2
    add_arc_arrow(dwg, f"M {cx-83} {cy-18} C {cx-54} {cy-96}, {cx+48} {cy-96}, {cx+86} {cy-35}", (cx + 86, cy - 35, 35), "#666666", 4)
    add_arc_arrow(dwg, f"M {cx+74} {cy+64} C {cx+26} {cy+106}, {cx-70} {cy+81}, {cx-87} {cy+12}", (cx - 87, cy + 12, -118), "#888888", 4)
    add_arc_arrow(dwg, f"M {cx-94} {cy+42} C {cx-36} {cy+16}, {cx+18} {cy-2}, {cx+68} {cy-42}", (cx + 68, cy - 42, -32), "#666666", 3.6)
    pts = [(cx - 67, cy + 44), (cx - 36, cy + 33), (cx - 6, cy + 18), (cx + 25, cy - 6), (cx + 57, cy + 53), (cx + 84, cy + 33)]
    for i, (pxi, pyi) in enumerate(pts):
        fill = "#606060" if i < 4 else "#b8b8b8"
        dwg.add(dwg.circle(center=(pxi, pyi), r=5.4, fill=fill))
    tx, ty = px + pw - 46, py + 94
    dwg.add(dwg.circle(center=(tx, ty), r=22, fill="none", stroke="#666666", stroke_width=3))
    dwg.add(dwg.line((tx - 30, ty), (tx + 30, ty), stroke="#666666", stroke_width=3, stroke_linecap="round"))
    dwg.add(dwg.line((tx, ty - 30), (tx, ty + 30), stroke="#666666", stroke_width=3, stroke_linecap="round"))
    dwg.add(dwg.circle(center=(tx, ty), r=5, fill="#666666"))
    add_info_box(
        dwg,
        x + 18,
        y + 330,
        w - 36,
        135,
        [
            "sample candidate λ",
            "generate target stream τ",
            "rollout with frozen πθ",
            "select τ*",
        ],
        bullet="square",
        color=TEXT,
        size=14,
    )
    add_info_box(dwg, x + 18, y + 485, w - 36, 73, ["optimize target stream,", "not actuator sequence"], bullet=None, color=TEXT, size=15)


def draw_policy(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    px, py = x + 18, y + 83
    pw, ph = w - 36, 224
    add_plot_box(dwg, px, py, pw, ph)
    layer_x = [px + 40, px + 120, px + 200]
    ys = [
        [py + 45, py + 89, py + 133, py + 177],
        [py + 45, py + 89, py + 133, py + 177],
        [py + 67, py + 112, py + 157],
    ]
    for ya in ys[0]:
        for yb in ys[1]:
            dwg.add(dwg.line((layer_x[0], ya), (layer_x[1], yb), stroke=LIGHT_LINE, stroke_width=1))
    for ya in ys[1]:
        for yb in ys[2]:
            dwg.add(dwg.line((layer_x[1], ya), (layer_x[2], yb), stroke=LIGHT_LINE, stroke_width=1))
    for li, lx in enumerate(layer_x):
        for yy in ys[li]:
            dwg.add(dwg.circle(center=(lx, yy), r=10, fill="#8f8f8f", stroke="#777777", stroke_width=1))
    lx, ly = px + pw - 40, py + 100
    dwg.add(dwg.path(d=f"M {lx-18} {ly-4} L {lx-18} {ly-24} C {lx-18} {ly-50}, {lx+18} {ly-50}, {lx+18} {ly-24} L {lx+18} {ly-4}", fill="none", stroke="#666666", stroke_width=5, stroke_linecap="round"))
    rounded_rect(dwg, lx - 25, ly - 4, 50, 48, stroke="#666666", fill="#666666", sw=2, rx=7)
    dwg.add(dwg.circle(center=(lx, ly + 17), r=5, fill="#ffffff"))
    dwg.add(dwg.line((lx, ly + 20), (lx, ly + 30), stroke="#ffffff", stroke_width=3, stroke_linecap="round"))
    add_info_box(dwg, x + 18, y + 330, w - 36, 76, ["Inputs: target stream τ,", "state x_t"], bullet="square", color=TEXT, size=14)
    add_info_box(dwg, x + 18, y + 422, w - 36, 76, ["Policy: quaternion-", "structured πθ"], bullet="circle", color=TEXT, size=14)
    add_info_box(dwg, x + 18, y + 514, w - 36, 56, ["Outputs: actuator command u_t"], bullet=None, color=TEXT, size=14)


def draw_simulator(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    px, py = x + 18, y + 83
    pw, ph = w - 36, 224
    add_plot_box(dwg, px, py, pw, ph)
    # Ground grid.
    gy = py + ph - 52
    for i in range(7):
        xx = px + 18 + i * 38
        dwg.add(dwg.line((xx, gy), (xx + 36, py + ph - 17), stroke="#d6d6d6", stroke_width=1))
    for j in range(3):
        yy = gy + j * 16
        dwg.add(dwg.line((px + 17, yy), (px + pw - 17, yy), stroke="#d6d6d6", stroke_width=1))
    # Aircraft line art: stylized side/top hybrid, intentionally simple.
    g = dwg.g(transform=f"translate({px+137},{py+99}) rotate(-8) scale(0.92)")
    g.add(dwg.path(d="M -98 4 C -58 -26, 46 -29, 104 -5 C 62 4, 18 12, -58 25 C -84 29, -111 16, -98 4 Z", fill="none", stroke="#5f5f5f", stroke_width=3.1, stroke_linejoin="round"))
    g.add(dwg.path(d="M -18 3 L -91 -42 L -104 -41 L -47 22", fill="none", stroke="#5f5f5f", stroke_width=3.1, stroke_linejoin="round"))
    g.add(dwg.path(d="M 12 2 L 82 -34 L 96 -32 L 42 18", fill="none", stroke="#5f5f5f", stroke_width=3.1, stroke_linejoin="round"))
    g.add(dwg.path(d="M 58 -8 L 81 -65 L 94 -67 L 76 -4", fill="none", stroke="#5f5f5f", stroke_width=3.1, stroke_linejoin="round"))
    g.add(dwg.path(d="M 58 13 L 107 24 L 106 29 L 48 24", fill="none", stroke="#5f5f5f", stroke_width=3.1, stroke_linejoin="round"))
    g.add(dwg.ellipse(center=(-57, 29), r=(13, 17), fill="none", stroke="#5f5f5f", stroke_width=3))
    g.add(dwg.ellipse(center=(45, 29), r=(13, 17), fill="none", stroke="#5f5f5f", stroke_width=3))
    g.add(dwg.path(d="M -95 10 C -103 23, -96 34, -80 39", fill="none", stroke="#5f5f5f", stroke_width=3, stroke_linecap="round"))
    dwg.add(g)
    add_info_box(dwg, x + 18, y + 330, w - 36, 76, ["6-DOF F-16 dynamics"], bullet="square", color=TEXT, size=14)
    add_info_box(dwg, x + 18, y + 422, w - 36, 76, ["actuator command u_t"], bullet="square", color=TEXT, size=14)
    add_info_box(dwg, x + 18, y + 514, w - 36, 56, ["next state x_{t+1}"], bullet=None, color=TEXT, size=14)


def draw_metrics(dwg: svgwrite.Drawing, x: float, y: float, w: float, h: float) -> None:
    px, py = x + 18, y + 83
    pw, ph = w - 36, 224
    add_plot_box(dwg, px, py, pw, ph)
    # Bar chart.
    bx, by = px + 32, py + 31
    dwg.add(dwg.line((bx, by + 84), (bx + 78, by + 84), stroke="#777777", stroke_width=2))
    dwg.add(dwg.line((bx, by + 84), (bx, by), stroke="#777777", stroke_width=2))
    for i, bh in enumerate([24, 41, 58, 76]):
        dwg.add(dwg.rect(insert=(bx + 13 + i * 18, by + 84 - bh), size=(11, bh), fill="#8c8c8c"))
    dwg.add(dwg.line((bx + 94, by - 6), (bx + 94, by + 91), stroke=LIGHT_LINE, stroke_width=2, stroke_dasharray="7,6"))
    # Line chart.
    lx, ly = px + 119, py + 31
    dwg.add(dwg.line((lx, ly + 84), (lx + 93, ly + 84), stroke="#777777", stroke_width=2))
    poly = [(lx + 6, ly + 68), (lx + 22, ly + 50), (lx + 40, ly + 59), (lx + 58, ly + 43), (lx + 72, ly + 47), (lx + 88, ly + 22)]
    dwg.add(dwg.polyline(poly, fill="none", stroke="#777777", stroke_width=3))
    for p in poly:
        dwg.add(dwg.circle(center=p, r=4, fill="#777777"))
    # Gauge.
    gx, gy = px + pw / 2, py + 163
    dwg.add(dwg.path(d=f"M {gx-58} {gy+30} A 58 58 0 0 1 {gx+58} {gy+30}", fill="none", stroke="#777777", stroke_width=5))
    for a in [-160, -125, -90, -55, -20]:
        x1, y1 = rotate_point(50, 0, a)
        x2, y2 = rotate_point(58, 0, a)
        dwg.add(dwg.line((gx + x1, gy + 30 + y1), (gx + x2, gy + 30 + y2), stroke="#777777", stroke_width=2))
    nx, ny = rotate_point(34, 0, -62)
    dwg.add(dwg.line((gx, gy + 30), (gx + nx, gy + 30 + ny), stroke="#777777", stroke_width=4, stroke_linecap="round"))
    dwg.add(dwg.circle(center=(gx, gy + 30), r=8, fill="#777777"))
    add_info_box(dwg, x + 18, y + 330, w - 36, 64, ["trajectory geometry"], bullet="circle", color=TEXT, size=14)
    add_info_box(dwg, x + 18, y + 410, w - 36, 64, ["attitude / quaternion error"], bullet="circle", color=TEXT, size=14)
    add_info_box(dwg, x + 18, y + 490, w - 36, 64, ["energy and lift consistency"], bullet=None, color=TEXT, size=14)


def draw_target_icon(dwg: svgwrite.Drawing, cx: float, cy: float, color: str) -> None:
    dwg.add(dwg.circle(center=(cx, cy), r=31, fill="none", stroke=color, stroke_width=5, opacity=0.7))
    dwg.add(dwg.circle(center=(cx, cy), r=15, fill="none", stroke=color, stroke_width=4, opacity=0.7))
    dwg.add(dwg.circle(center=(cx, cy), r=4.5, fill=color))
    arrow(dwg, cx + 3, cy - 3, cx + 43, cy - 43, color, width=4)


def draw_wave_icon(dwg: svgwrite.Drawing, x: float, y: float, color: str) -> None:
    for i in range(3):
        yy = y + i * 20
        dwg.add(dwg.path(d=f"M {x} {yy} C {x+25} {yy-14}, {x+47} {yy+14}, {x+72} {yy} C {x+98} {yy-14}, {x+122} {yy+14}, {x+148} {yy}", fill="none", stroke=color, stroke_width=4, stroke_linecap="round", opacity=0.75))


def draw_check_icon(dwg: svgwrite.Drawing, cx: float, cy: float, color: str) -> None:
    dwg.add(dwg.circle(center=(cx, cy), r=36, fill="none", stroke=color, stroke_width=5, opacity=0.75))
    dwg.add(dwg.path(d=f"M {cx-17} {cy} L {cx-4} {cy+15} L {cx+23} {cy-18}", fill="none", stroke=color, stroke_width=6, stroke_linecap="round", stroke_linejoin="round"))


def draw_bottom_pipeline(dwg: svgwrite.Drawing) -> None:
    y, h = 735, 112
    xs = [35, 580, 1125]
    bw = 440
    colors = [BLUE, ORANGE, GREEN]
    titles = [
        ["Quaternion-based flight skill"],
        ["Executable target streams", "+ RH-TSO"],
        ["Accurate agile maneuver", "execution"],
    ]
    for i, (x, color, title_lines) in enumerate(zip(xs, colors, titles)):
        rounded_rect(dwg, x, y, bw, h, stroke=color, fill="#ffffff", sw=2.1, rx=10)
        dwg.add(dwg.circle(center=(x + 47, y + 56), r=27, fill=color))
        text(dwg, str(i + 1), x + 47, y + 66, size=29, fill="#ffffff", weight="700", anchor="middle")
        for j, title in enumerate(title_lines):
            text(dwg, title, x + 95, y + 43 + j * 22, size=18, fill=TEXT, weight="700")
        if i == 0:
            text(dwg, "quaternion target tracking", x + 95, y + 75, size=15, fill=MUTED)
            draw_target_icon(dwg, x + bw - 72, y + 57, color)
        elif i == 1:
            text(dwg, "closed-loop selection", x + 95, y + 91, size=15, fill=MUTED)
            draw_wave_icon(dwg, x + bw - 150, y + 43, color)
        else:
            text(dwg, "geometry- and energy-consistent", x + 95, y + 91, size=15, fill=MUTED)
            draw_check_icon(dwg, x + bw - 72, y + 57, color)
    arrow(dwg, xs[0] + bw + 24, y + h / 2, xs[1] - 23, y + h / 2, BLUE, width=7)
    arrow(dwg, xs[1] + bw + 24, y + h / 2, xs[2] - 23, y + h / 2, ORANGE, width=7)


def build_svg() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if SRC_DRAFT.exists():
        shutil.copyfile(SRC_DRAFT, DRAFT_COPY)

    dwg = svgwrite.Drawing(str(SVG_OUT), size=(f"{W}px", f"{H}px"), viewBox=f"0 0 {W} {H}", profile="full")
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill="#ffffff"))

    panel_y, panel_h = 55, 625
    margin, gap = 34, 26
    panel_w = (W - 2 * margin - 4 * gap) / 5
    panels = [
        ("Reference Maneuver", BLUE, draw_reference),
        ("RH-TSO", ORANGE, draw_rhtso),
        ("Frozen RL Policy", DARK, draw_policy),
        ("High-Fidelity Simulator", BLUE, draw_simulator),
        ("Geometry-Aware Metrics", GREEN, draw_metrics),
    ]
    xs = [margin + i * (panel_w + gap) for i in range(5)]
    for x, (title, color, drawer) in zip(xs, panels):
        add_panel(dwg, x, panel_y, panel_w, panel_h, title, color)
        drawer(dwg, x, panel_y, panel_w, panel_h)

    arrow_y = panel_y + 315
    arrow_colors = [BLUE, ORANGE, DARK, GREEN]
    for i in range(4):
        arrow(dwg, xs[i] + panel_w + 4, arrow_y, xs[i + 1] - 4, arrow_y, arrow_colors[i], width=6)

    draw_bottom_pipeline(dwg)
    dwg.save()


def convert_outputs() -> None:
    inkscape = shutil.which("inkscape")
    if inkscape:
        subprocess.run(
            [inkscape, str(SVG_OUT), "--export-type=pdf", f"--export-filename={PDF_OUT}"],
            check=True,
        )
        subprocess.run(
            [
                inkscape,
                str(SVG_OUT),
                "--export-type=png",
                f"--export-filename={PNG_OUT}",
                "--export-width=1600",
                "--export-height=900",
            ],
            check=True,
        )
    else:
        import cairosvg

        cairosvg.svg2pdf(url=str(SVG_OUT), write_to=str(PDF_OUT), output_width=W, output_height=H)
        cairosvg.svg2png(url=str(SVG_OUT), write_to=str(PNG_OUT), output_width=W, output_height=H)


def run_checks() -> None:
    svg_text = SVG_OUT.read_text(encoding="utf-8").lower()
    if "<image" in svg_text:
        raise RuntimeError("SVG contains a raster <image> tag, which is not allowed.")
    if not PDF_OUT.exists():
        raise RuntimeError(f"PDF was not generated: {PDF_OUT}")
    if not PNG_OUT.exists():
        raise RuntimeError(f"PNG preview was not generated: {PNG_OUT}")


def main() -> None:
    build_svg()
    convert_outputs()
    run_checks()
    print("Generated overview figure:")
    print(f"  draft reference: {DRAFT_COPY}")
    print(f"  SVG: {SVG_OUT}")
    print(f"  PDF: {PDF_OUT}")
    print(f"  PNG preview: {PNG_OUT}")


if __name__ == "__main__":
    main()
