"""
Génère du texte avec un modèle entraîné.

Usage :
    python generate.py --checkpoint_dir checkpoints --prompt "La France est" --tokens 300
    python generate.py --checkpoint_dir checkpoints --prompt "La France est" --seed 42   # reproductible
"""

import argparse
import json
import os

import torch

from model import MiniGPT, GPTConfig
from tokenizer import BPETokenizer
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixe la graine aléatoire pour un échantillonnage reproductible. "
                             "Par défaut (non fourni), chaque appel produit un texte différent "
                             "même à paramètres identiques -- utile pour comparer proprement "
                             "l'effet d'un seul paramètre (ex: température) sans bruit d'échantillonnage.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()

    with open(os.path.join(args.checkpoint_dir, "config.json")) as f:
        config_dict = json.load(f)
        # Filtrer pour ne garder que les champs attendus par GPTConfig
        filtered_config = {
            k: v for k, v in config_dict.items()
            if k in ["vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout"]
        }
        config = GPTConfig(**filtered_config)

    tokenizer = BPETokenizer.load(os.path.join(args.checkpoint_dir, "tokenizer.json"))

    model = MiniGPT(config).to(device)
    state_dict = torch.load(os.path.join(args.checkpoint_dir, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = tokenizer.decode(out[0].tolist())

    print(text)


if __name__ == "__main__":
    main()