"""
Trace la courbe de loss (train + val) à partir de history.json.
Génère une image prête à poster sur LinkedIn.

Usage :
    python plot_loss.py --checkpoint_dir checkpoints
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--out", type=str, default="loss_curve.png")
    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_dir, "history.json")) as f:
        history = json.load(f)

    steps = [h["step"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_losses, label="train loss", linewidth=2)
    plt.plot(steps, val_losses, label="val loss", linewidth=2, linestyle="--")
    plt.xlabel("Step d'entraînement")
    plt.ylabel("Loss (cross-entropy)")
    plt.title("Entraînement du MiniGPT from scratch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Graphique sauvegardé : {args.out}")


if __name__ == "__main__":
    main()