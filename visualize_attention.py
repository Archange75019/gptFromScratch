"""
Visualise la matrice d'attention d'une tête donnée, pour un prompt donné.
Excellent contenu visuel pour un post LinkedIn : ça montre concrètement
"ce que regarde" le modèle pour prédire chaque token.

Usage :
    python visualize_attention.py --checkpoint_dir checkpoints --prompt "La France est un pays" --layer 0 --head 0
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from model import MiniGPT, GPTConfig
from tokenizer import BPETokenizer
from train import get_device


def get_attention_weights(model, idx, layer_idx):
    """Poids d'attention (B, n_head, T, T) de la couche `layer_idx`.

    On fait tourner les blocs précédents normalement, puis on demande au bloc
    cible ses poids d'attention (RoPE inclus) : plus de calcul recopié à la main.
    """
    x = model.token_embedding(idx)
    for i in range(layer_idx):
        x = model.blocks[i](x, model.rope_cos, model.rope_sin)
    block = model.blocks[layer_idx]
    return block.attn.attention_weights(block.ln1(x), model.rope_cos, model.rope_sin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--prompt", type=str, default="La France est un pays")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--out", type=str, default="attention_heatmap.png")
    args = parser.parse_args()

    device = get_device()
    with open(os.path.join(args.checkpoint_dir, "config.json")) as f:
        config = GPTConfig(**json.load(f))
    tokenizer = BPETokenizer.load(os.path.join(args.checkpoint_dir, "tokenizer.json"))

    model = MiniGPT(config).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pt"), map_location=device))
    model.eval()

    token_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        att_weights = get_attention_weights(model, idx, args.layer)

    weights = att_weights[0, args.head].cpu().numpy()  # (T, T)
    # Contrairement à un tokenizer character-level, un token BPE peut
    # représenter un mot entier ou un sous-mot -- on récupère son texte
    # exact (pas juste son ID) pour des labels de heatmap lisibles.
    tokens = [tokenizer.token_str(t) for t in token_ids]

    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    plt.yticks(range(len(tokens)), tokens, fontsize=8)
    plt.xlabel("Token regardé (clé)")
    plt.ylabel("Token qui regarde (requête)")
    plt.title(f"Attention -- couche {args.layer}, tête {args.head}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Heatmap sauvegardée : {args.out}")


if __name__ == "__main__":
    main()