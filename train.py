"""
Boucle d'entraînement du MiniGPT sur un corpus pré-tokenisé (Wikipédia FR).

Usage :
    python train.py --data_dir data_prepared --steps 5000

`data_dir` doit avoir été produit par prepare_data.py (train.bin, val.bin,
tokenizer.json, meta.json). Les données sont lues via numpy.memmap : chaque
batch ne relit que les quelques milliers d'entiers dont il a besoin, jamais
le corpus entier -- ça passe à l'échelle sur plusieurs Go sans changer la
RAM utilisée.

Le script :
  1. charge les données (memmap sur train.bin/val.bin)
  2. entraîne le modèle en suivant la loss (utile pour l'article LinkedIn :
     capture les courbes avec matplotlib, cf. plot_loss.py)
  3. sauvegarde le modèle + le tokenizer pour la génération ensuite
"""

import argparse
import time
import json
import os
import shutil

import numpy as np
import torch
from tqdm import trange

from model import MiniGPT, GPTConfig
from tokenizer import BPETokenizer


class LiveLossWindow:
    """
    Fenêtre matplotlib qui se met à jour en direct pendant l'entraînement.

    Utile pour VOIR que l'entraînement avance (et pas juste s'y fier via les
    logs texte) : la loss train/val se trace au fur et à mesure, sans
    attendre la fin pour lancer plot_loss.py.

    S'auto-désactive proprement si aucun affichage n'est disponible (ex :
    serveur distant sans X11 / SSH sans -X) pour ne jamais faire planter
    l'entraînement.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.ok = False
        if not enabled:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
            (self.train_line,) = self.ax.plot([], [], label="train loss", linewidth=2)
            (self.val_line,) = self.ax.plot([], [], label="val loss", linewidth=2, linestyle="--")
            self.ax.set_xlabel("step")
            self.ax.set_ylabel("loss (cross-entropy)")
            self.ax.set_title("Entraînement en cours...")
            self.ax.legend()
            self.ax.grid(alpha=0.3)
            self.fig.tight_layout()
            self.fig.canvas.manager.set_window_title("MiniGPT - suivi de l'entraînement")
            self.plt.show(block=False)
            self.ok = True
        except Exception as e:
            print(f"[live plot désactivé : {e}] -- pas d'affichage disponible, "
                  "l'entraînement continue normalement (logs texte + history.json).")

    def update(self, history):
        if not self.ok:
            return
        steps = [h["step"] for h in history]
        self.train_line.set_data(steps, [h["train_loss"] for h in history])
        self.val_line.set_data(steps, [h["val_loss"] for h in history])
        self.ax.relim()
        self.ax.autoscale_view()
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            # la fenêtre a pu être fermée manuellement par l'utilisateur
            self.ok = False

    def close(self):
        if self.ok:
            self.plt.ioff()


class BinDataset:
    """
    Vue sur un fichier .bin de tokens (uint16) produit par prepare_data.py.

    Le fichier n'est JAMAIS chargé entier en RAM : `open()` retourne un
    numpy.memmap (mapping mémoire à la demande, pas une lecture complète).
    On rouvre ce memmap une fois PAR APPEL À get_batch (donc une fois par
    step, pas une fois par exemple du batch) : assez rare pour ne pas
    coûter cher, assez fréquent pour éviter le risque de fuite mémoire
    documenté sur de très longues boucles avec un memmap gardé ouvert en
    permanence.
    """

    def __init__(self, path: str):
        self.path = path
        self.dtype = np.uint16
        self._len = os.path.getsize(path) // np.dtype(self.dtype).itemsize

    def __len__(self):
        return self._len

    def open(self) -> np.ndarray:
        return np.memmap(self.path, dtype=self.dtype, mode="r")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # sur les machines Intel avec PyTorch + intel-extension-for-pytorch,
    # le device XPU (Arc GPU) peut être disponible :
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Mac Apple Silicon (M1/M2/M3...) :
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data: BinDataset, block_size, batch_size, device):
    """Un SEUL memmap ouvert pour tout le batch, réutilisé pour les 64
    tranches (au lieu d'un memmap par exemple : ouvrir un memmap a un coût
    -- réel, même s'il est petit -- et le refaire 128 fois par step finit
    par s'accumuler, en particulier sous Windows où le mapping/démapping
    mémoire répété peut ralentir progressivement tout le processus au fil
    de l'entraînement).
    """
    arr = data.open()
    ix = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(arr[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dossier produit par prepare_data.py (train.bin/val.bin/tokenizer.json).")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Reprend un entraînement interrompu depuis --out_dir : recharge "
                             "modèle + optimiseur + historique, et continue jusqu'à --steps "
                             "(l'architecture est relue depuis le config.json existant, les "
                             "--n_layer/--n_head/--n_embd/--block_size passés en argument sont ignorés).")
    parser.add_argument("--no_live_plot", action="store_true",
                        help="Désactive la fenêtre matplotlib live (utile en SSH sans affichage).")
    args = parser.parse_args()

    device = get_device()
    print(f"Device utilisé : {device}")

    if args.resume and not os.path.exists(os.path.join(args.out_dir, "model.pt")):
        raise SystemExit(f"--resume demandé mais aucun checkpoint trouvé dans {args.out_dir}/")

    print(f"Lecture depuis {args.data_dir}/ (memmap, tokenizer BPE)")
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    train_data = BinDataset(os.path.join(args.data_dir, "train.bin"))
    val_data = BinDataset(os.path.join(args.data_dir, "val.bin"))
    print(f"  {meta['n_train_tokens']:,} tokens train / {meta['n_val_tokens']:,} tokens val "
          f"/ vocab_size={vocab_size}")
    tokenizer_src_path = os.path.join(args.data_dir, "tokenizer.json")

    if args.resume:
        # Architecture relue depuis le checkpoint existant : elle DOIT
        # matcher les poids sauvegardés, donc on ignore les --n_layer/
        # --n_head/--n_embd/--block_size éventuellement passés en argument.
        with open(os.path.join(args.out_dir, "config.json")) as f:
            prev_config_dict = json.load(f)
        config = GPTConfig(**prev_config_dict)
        if config.vocab_size != vocab_size:
            raise SystemExit(
                f"Le corpus fourni a un vocab_size={vocab_size}, différent de celui du "
                f"checkpoint ({config.vocab_size}). --resume nécessite le même --data_dir "
                "que l'entraînement initial."
            )
    else:
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )

    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    history = []
    start_step = 0
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pt"), map_location=device))
        optimizer_path = os.path.join(args.out_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        history_path = os.path.join(args.out_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            if history:
                start_step = history[-1]["step"] + 1
        print(f"Reprise depuis le step {start_step} (checkpoint : {args.out_dir}/model.pt)")
    else:
        shutil.copyfile(tokenizer_src_path, os.path.join(args.out_dir, "tokenizer.json"))
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config.__dict__, f)

    live_window = LiveLossWindow(enabled=not args.no_live_plot)
    t0 = time.time()

    # trange = tqdm(range(...)) : donne un feedback à CHAQUE step (steps/s,
    # ETA, loss courante), quasi gratuit puisque la loss du batch est de
    # toute façon déjà calculée. C'est ça qui évite l'impression de blocage
    # pendant les eval_interval steps entre deux évaluations complètes.
    # range(start_step, ...) : en mode --resume, on continue là où le
    # dernier checkpoint s'est arrêté plutôt que de repartir de 0.
    progress = trange(start_step, args.steps + 1, desc="Entraînement", unit="step")
    for step in progress:
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - t0
            progress.write(
                f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s"
            )
            history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"]})

            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.out_dir, "optimizer.pt"))
            with open(os.path.join(args.out_dir, "history.json"), "w") as f:
                json.dump(history, f)

            live_window.update(history)

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")  # affiché en continu, coût ~nul

    live_window.close()
    print("Entraînement terminé. Checkpoint dans:", args.out_dir)


if __name__ == "__main__":
    main()