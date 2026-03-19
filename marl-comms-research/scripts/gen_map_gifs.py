"""
Generate clean map preview GIFs for PowerPoint slides.
Dark theme: navy floor, WHITE obstacles, grey grid, green pursuers, orange evader.
Run from repo root: python marl-comms-research/scripts/gen_map_gifs.py
"""
import sys
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.fixed_pursuit import MAP_SPECS

# ── Colors ────────────────────────────────────────────────────────────────────
C_FLOOR    = (30,  32,  48)    # dark navy
C_WALL     = (220, 220, 220)   # bright light grey / near-white (visible!)
C_GRID     = (65,  68,  85)    # subtle grey grid lines
C_PURSUER  = (80,  220, 120)   # green
C_EVADER   = (255, 140,  40)   # orange
C_OUTLINE_P = (160, 255, 180)  # light green outline
C_OUTLINE_E = (255, 200, 100)  # light orange outline

CELL   = 32   # pixels per cell
RADIUS = 9    # agent circle radius
FRAMES = 18
DURATION = 300  # ms per frame

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def draw_frame(grid, pursuers, evader):
    size = grid.shape[0]
    img = Image.new("RGB", (size * CELL, size * CELL), C_FLOOR)
    draw = ImageDraw.Draw(img)

    # Fill walls
    for x in range(size):
        for y in range(size):
            if grid[x, y] == -1:
                px, py = x * CELL, y * CELL
                draw.rectangle([px, py, px + CELL - 1, py + CELL - 1], fill=C_WALL)

    # Grid lines
    for i in range(size + 1):
        draw.line([(i * CELL, 0), (i * CELL, size * CELL)], fill=C_GRID, width=1)
        draw.line([(0, i * CELL), (size * CELL, i * CELL)], fill=C_GRID, width=1)

    # Agents — draw at (col=x, row=y) → pixel centre (x*CELL + CELL//2, y*CELL + CELL//2)
    def draw_agent(col, row, fill, outline):
        cx = col * CELL + CELL // 2
        cy = row * CELL + CELL // 2
        r = RADIUS
        draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2], fill=outline)
        draw.ellipse([cx-r,   cy-r,   cx+r,   cy+r  ], fill=fill)

    for (col, row) in pursuers:
        draw_agent(col, row, C_PURSUER, C_OUTLINE_P)

    draw_agent(evader[0], evader[1], C_EVADER, C_OUTLINE_E)

    return img


def random_step(pos, grid):
    """Move one step in a random valid direction (or stay)."""
    col, row = pos
    size = grid.shape[0]
    options = [(col, row)]
    for dc, dr in [(0,1),(0,-1),(1,0),(-1,0)]:
        nc, nr = col+dc, row+dr
        if 0 <= nc < size and 0 <= nr < size and grid[nc, nr] != -1:
            options.append((nc, nr))
    return random.choice(options)


def make_gif(map_name, out_path):
    spec = MAP_SPECS[map_name]
    grid = spec.grid
    pursuers = list(spec.pursuer_starts)
    evader   = list(spec.evader_starts[0])

    random.seed(42)
    frames = []
    for _ in range(FRAMES):
        frames.append(draw_frame(grid, pursuers, evader))
        pursuers = [random_step(p, grid) for p in pursuers]
        evader   = list(random_step(tuple(evader), grid))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=DURATION,
        optimize=False,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name in ["easy_open", "center_block", "split_barrier"]:
        make_gif(name, OUT_DIR / f"ppt_map_{name}.gif")
    print("Done.")
