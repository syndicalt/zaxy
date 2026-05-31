#!/usr/bin/env python3
"""Generate scripted Zaxy 1.0 release demo media."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "docs" / "media"
FRAME_DIR = MEDIA_DIR / "zaxy-collaborate-demo-frames"
MP4_PATH = MEDIA_DIR / "zaxy-collaborate-demo.mp4"
GIF_PATH = MEDIA_DIR / "zaxy-collaborate-demo.gif"
W, H = 1600, 900


def main() -> int:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if FRAME_DIR.exists():
        shutil.rmtree(FRAME_DIR)
    FRAME_DIR.mkdir(parents=True)

    frames = [
        (
            "Zaxy 1.0",
            "Coordinator Memory for Agent Teams",
            ["zaxy init", "memory_bootstrap", "memory_checkout"],
            "Local Eventloom memory starts with cited bootstrap guidance.",
        ),
        (
            "Parent Mission",
            "One durable history for the project",
            ["coordination_start", "mission: release-readiness", "accepted state: empty"],
            "The coordinator owns project truth. Workers investigate in isolation.",
        ),
        (
            "Worker Findings",
            "Parallel agents report evidence, not guesses",
            ["worker-api: finding + citation", "worker-docs: finding + citation", "worker-ui: finding + citation"],
            "Each worker keeps its own Eventloom log and submits reviewable claims.",
        ),
        (
            "Review and Promotion",
            "Conflicts are visible before they become memory",
            ["accepted", "deferred", "conflict detected", "approval packet"],
            "Only reviewed findings are promoted into the parent mission state.",
        ),
        (
            "Memory Checkout",
            "The next agent turn receives cited, accepted context",
            ["answerability: answer_from_memory", "citations: present", "feedback: requested"],
            "Zaxy turns replayable history into prompt-ready working memory.",
        ),
    ]

    for index, frame in enumerate(frames):
        draw_frame(index, *frame)

    run_ffmpeg()
    print(f"Wrote {MP4_PATH.relative_to(ROOT)}")
    print(f"Wrote {GIF_PATH.relative_to(ROOT)}")
    return 0


def draw_frame(index: int, title: str, subtitle: str, items: list[str], footer: str) -> None:
    image = Image.new("RGB", (W, H), "#071013")
    draw = ImageDraw.Draw(image)
    title_font = font(74, bold=True)
    subtitle_font = font(32)
    body_font = font(31)
    small_font = font(26)

    draw_grid(draw)
    next_y = draw_wrapped_text(
        draw,
        title,
        (92, 78),
        title_font,
        "#f7faf6",
        max_width=600,
        line_gap=8,
    )
    draw_wrapped_text(
        draw,
        subtitle,
        (98, next_y + 10),
        subtitle_font,
        "#b8c8c3",
        max_width=590,
        line_gap=6,
    )

    timeline_y = 690
    for i in range(5):
        x = 150 + i * 280
        color = "#39f3ca" if i <= index else "#35545a"
        draw.line((150, timeline_y, 1270, timeline_y), fill="#24525b", width=5)
        draw.ellipse((x - 22, timeline_y - 22, x + 22, timeline_y + 22), fill=color)
        draw.text((x - 8, timeline_y - 13), str(i + 1), font=small_font, fill="#071013")

    draw.rounded_rectangle((735, 140, 1440, 600), radius=28, outline="#2be7bd", width=3, fill="#0b171b")
    draw.text((780, 185), "Eventloom -> Graph -> Checkout", font=body_font, fill="#d9f7ef")

    y = 250
    for item_index, item in enumerate(items):
        accent = ["#2be7bd", "#4cc9f0", "#9df65f", "#f4d35e"][item_index % 4]
        draw.rounded_rectangle((780, y, 1388, y + 66), radius=16, fill="#12242a", outline="#264f57")
        draw.ellipse((806, y + 20, 832, y + 46), fill=accent)
        draw.text((852, y + 17), item, font=small_font, fill="#edf7f4")
        y += 88

    draw.rounded_rectangle((92, 760, 1440, 832), radius=20, fill="#0e1d21", outline="#274c53")
    draw.text((124, 782), footer, font=small_font, fill="#cfe3dd")

    image.save(FRAME_DIR / f"frame-{index:02d}.png")


def draw_grid(draw: ImageDraw.ImageDraw) -> None:
    for x in range(0, W, 80):
        draw.line((x, 0, x, H), fill="#10272c", width=1)
    for y in range(0, H, 80):
        draw.line((0, y, W, y), fill="#10272c", width=1)
    draw.ellipse((1110, 220, 1350, 460), outline="#2be7bd", width=4)
    center = (1230, 340)
    nodes = [(1160, 300), (1230, 260), (1300, 300), (1300, 390), (1230, 430), (1160, 390)]
    for node in nodes:
        draw.line((center[0], center[1], node[0], node[1]), fill="#9df65f", width=3)
        draw.ellipse((node[0] - 12, node[1] - 12, node[0] + 12, node[1] + 12), fill="#9df65f")
    draw.ellipse((center[0] - 16, center[1] - 16, center[0] + 16, center[1] + 16), fill="#f7faf6")


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    text_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
    *,
    max_width: int,
    line_gap: int,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=text_font)
        if current and bbox[2] - bbox[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def run_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate release demo media")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "0.7",
            "-i",
            str(FRAME_DIR / "frame-%02d.png"),
            "-vf",
            "format=yuv420p",
            str(MP4_PATH),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "0.7",
            "-i",
            str(FRAME_DIR / "frame-%02d.png"),
            "-vf",
            "scale=960:-1:flags=lanczos",
            str(GIF_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
