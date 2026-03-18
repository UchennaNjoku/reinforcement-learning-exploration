"""Rollout visualization — renders a greedy episode and saves as a GIF.

For comm models, overlays the current message each agent is sending on every frame.
For baseline models, renders without message overlay.

Outputs a GIF to the specified path.

Usage:
    # Comm-4 on easy_open
    python render_rollout.py --checkpoint results/comm4_v2/checkpoints/comm4_ep004000.pt

    # Comm-16 on split_barrier
    python render_rollout.py --checkpoint results/comm16_v2/checkpoints/comm16_ep003500.pt --map split_barrier

    # Baseline
    python render_rollout.py --checkpoint results/baseline_v3/checkpoints/baseline_ep004000.pt

    # Save to explicit path
    python render_rollout.py --checkpoint results/comm4_v2/checkpoints/comm4_ep004000.pt --out results/rollout_comm4.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from eval import load_checkpoint, greedy_baseline, greedy_comm
from envs import make_fixed_pursuit_env
from train_comm import zero_messages, AGENT_IDS, N_AGENTS


AGENT_COLORS = {
    "pursuer_0": (228, 26, 28),    # red
    "pursuer_1": (55,  126, 184),  # blue
    "pursuer_2": (77,  175, 74),   # green
}
AGENT_SHORT = {"pursuer_0": "P0", "pursuer_1": "P1", "pursuer_2": "P2"}


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    step: int,
    messages: dict[str, int] | None,
    captured: bool,
    vocab_size: int | None,
) -> "PIL.Image.Image":
    """Add step counter and message overlay to an rgb_array frame."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame).convert("RGB")
    W, H = img.size

    panel_h = 48 if messages else 22
    canvas = Image.new("RGB", (W, H + panel_h), (30, 30, 30))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font_sm = ImageFont.load_default()
        font_lg = font_sm

    status = f"Step {step}" + ("  ★ CAPTURED!" if captured else "")
    draw.text((4, H + 3), status, fill=(220, 220, 220), font=font_lg)

    if messages is not None and vocab_size is not None:
        badge_w = W // 3
        for i, (agent, sym) in enumerate(sorted(messages.items())):
            x = i * badge_w
            y = H + 22
            color = AGENT_COLORS.get(agent, (200, 200, 200))
            draw.rectangle([x + 2, y, x + badge_w - 2, y + 20], fill=(*color, 200))
            label = f"{AGENT_SHORT[agent]}: sym {sym}/{vocab_size - 1}"
            draw.text((x + 5, y + 3), label, fill=(255, 255, 255), font=font_sm)

    return canvas


# ---------------------------------------------------------------------------
# Run one rollout, collect frames
# ---------------------------------------------------------------------------

def run_rollout(
    checkpoint_path: str,
    map_name: str | None,
    seed: int,
    max_steps: int,
) -> tuple[list, dict]:
    """Run one greedy episode and return (frames, info_dict)."""
    import torch

    device = torch.device("cpu")   # render on CPU for simplicity
    model, ckpt, is_comm = load_checkpoint(checkpoint_path, device)

    saved = ckpt.get("args", {})
    eval_map = map_name or saved.get("map", "easy_open")
    eval_n_catch = saved.get("n_catch", 1)

    if map_name and map_name != saved.get("map"):
        print(f"  NOTE: rendering on map='{map_name}' (trained on '{saved.get('map')}') — cross-map.")

    vocab_size = model.vocab_size if is_comm else None
    print(f"Model: {'comm' if is_comm else 'baseline'}"
          + (f"  vocab_size={vocab_size}" if is_comm else "")
          + f"  |  map={eval_map}")

    env = make_fixed_pursuit_env(
        map_name=eval_map,
        n_catch=eval_n_catch,
        surround=False,
        render_mode="rgb_array",
    )

    obs, _ = env.reset(seed=seed)

    if is_comm:
        prev_messages = zero_messages(vocab_size)

    frames = []
    step = 0
    captured = False
    current_msg_idx: dict[str, int] | None = {} if is_comm else None

    while step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(annotate_frame(
                frame, step,
                current_msg_idx if is_comm else None,
                captured, vocab_size,
            ))

        if is_comm:
            actions, new_messages = greedy_comm(model, obs, prev_messages, device)
            current_msg_idx = {
                a: int(np.argmax(new_messages[a])) for a in new_messages
            }
            prev_messages = new_messages
        else:
            actions = greedy_baseline(model, obs, device)

        next_obs, _, terminations, truncations, infos = env.step(actions)
        step += 1

        for info in infos.values():
            if info.get("evaders_remaining", 1) == 0:
                captured = True

        done = any(terminations.values()) or any(truncations.values()) or not env.agents
        obs = next_obs

        if done:
            frame = env.render()
            if frame is not None:
                frames.append(annotate_frame(
                    frame, step,
                    current_msg_idx if is_comm else None,
                    captured, vocab_size,
                ))
            break

    env.close()
    return frames, {"captured": captured, "steps": step, "map": eval_map}


# ---------------------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------------------

def save_gif(frames: list, out_path: Path, fps: int) -> None:
    if not frames:
        print("No frames to save.")
        return

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF saved → {out_path}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a greedy rollout as a GIF")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--map",        default=None,
                   choices=["easy_open", "center_block", "split_barrier", "large_split"],
                   help="Map to render on (default: checkpoint's training map)")
    p.add_argument("--seed",       type=int, default=99)
    p.add_argument("--fps",        type=int, default=8,
                   help="Frames per second in output GIF")
    p.add_argument("--max-steps",  type=int, default=300)
    p.add_argument("--out",        default=None,
                   help="Output GIF path (default: auto-named in results/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if args.out:
        out_path = Path(args.out)
    else:
        stem = ckpt_path.parent.parent.name  # e.g. comm4_v2
        map_tag = args.map or "default_map"
        out_path = Path("results") / f"rollout_{stem}_{map_tag}.gif"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running rollout: {ckpt_path.name}")
    frames, info = run_rollout(
        checkpoint_path=str(ckpt_path),
        map_name=args.map,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    print(f"Episode: captured={info['captured']}, steps={info['steps']}, map={info['map']}")
    save_gif(frames, out_path, fps=args.fps)
