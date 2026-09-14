"""
Prépare le corpus Wikipédia (potentiellement plusieurs centaines de Mo,
voire plusieurs Go) pour l'entraînement -- SANS jamais le charger
entièrement en RAM sous forme de string.

Ce que fait ce script :
  1. Entraîne un tokenizer BPE (librairie `tokenizers`) sur un ÉCHANTILLON
     du corpus (quelques centaines de Mo suffisent largement pour apprendre
     un vocabulaire stable -- pas besoin de tout voir).
  2. Relit le corpus ENTIER par blocs de quelques Mo, encode chaque bloc,
     et écrit les IDs de tokens (uint16) directement sur disque au fur et
     à mesure, dans train.bin / val.bin. À aucun moment le texte complet
     ni la liste complète des tokens ne sont gardés en mémoire.

Pourquoi uint16 : avec un vocab_size < 65536, chaque token tient sur 2
octets, et un token BPE couvre en moyenne ~3-4 caractères -- au total,
pour le même texte source, ça représente un ordre de grandeur en moins
d'octets à manipuler qu'un encodage naïf en mémoire.

train.py lit ensuite ces fichiers .bin via numpy.memmap : chaque batch ne
relit que les quelques milliers d'entiers dont il a besoin, jamais le
fichier entier.

Usage :
    python prepare_data.py --input corpus_wikipedia_fr.txt --out_dir data_prepared --vocab_size 8000

Puis :
    python train.py --data_dir data_prepared --steps 5000
"""

import argparse
import json
import os

import numpy as np

from tokenizer import BPETokenizer


def iter_chunks(path: str, chunk_chars: int = 2_000_000, hard_cap_factor: int = 5):
    """Lit `path` par blocs d'environ `chunk_chars` caractères, en coupant
    toujours sur une fin de ligne (jamais au milieu d'un mot/paragraphe).

    Sécurité : si une "ligne" fait plus de hard_cap_factor * chunk_chars
    (fichier sans retours à la ligne, cas pathologique), on force quand
    même une coupe pour ne jamais laisser le buffer grossir sans limite.
    """
    buf = ""
    hard_cap = chunk_chars * hard_cap_factor
    with open(path, "r", encoding="utf-8") as f:
        while True:
            data = f.read(chunk_chars)
            if not data:
                if buf:
                    yield buf
                break
            buf += data
            cut = buf.rfind("\n")
            if cut == -1:
                if len(buf) > hard_cap:
                    yield buf
                    buf = ""
                continue
            yield buf[: cut + 1]
            buf = buf[cut + 1:]


def build_bpe_sample(input_path: str, out_dir: str, max_mb: int) -> str:
    """Écrit un échantillon du corpus (les premiers max_mb Mo) dans un
    fichier temporaire, utilisé uniquement pour entraîner le tokenizer.
    """
    sample_path = os.path.join(out_dir, "_bpe_train_sample.txt")
    max_bytes = max_mb * 1_000_000
    written = 0
    with open(sample_path, "w", encoding="utf-8") as out:
        for chunk in iter_chunks(input_path):
            out.write(chunk)
            written += len(chunk.encode("utf-8"))
            if written >= max_bytes:
                break
    return sample_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Fichier texte brut source (le corpus Wikipédia téléchargé)")
    parser.add_argument("--out_dir", type=str, default="data_prepared")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction de chaque bloc envoyée en validation (0.1 = 10%%). "
                             "Appliquée à l'intérieur de chaque bloc (pas en alternant des blocs "
                             "entiers) pour que train/val restent bien répartis même sur un petit corpus.")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Chemin vers un tokenizer.json déjà entraîné à réutiliser "
                             "(sinon on en entraîne un nouveau sur le corpus fourni)")
    parser.add_argument("--bpe_sample_mb", type=int, default=200,
                        help="Taille (Mo) de l'échantillon utilisé pour ENTRAÎNER le tokenizer BPE "
                             "(l'entraînement BPE lui-même doit voir son échantillon en mémoire, "
                             "donc on le limite -- inutile de toute façon de lui montrer des Go entiers)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Tokenizer BPE
    # ------------------------------------------------------------------
    if args.tokenizer:
        print(f"Chargement du tokenizer existant : {args.tokenizer}")
        tok = BPETokenizer.load(args.tokenizer)
    else:
        print(f"Entraînement d'un tokenizer BPE (vocab_size={args.vocab_size}) "
              f"sur un échantillon de ~{args.bpe_sample_mb} Mo...")
        sample_path = build_bpe_sample(args.input, args.out_dir, args.bpe_sample_mb)
        tok = BPETokenizer.train([sample_path], vocab_size=args.vocab_size)
        os.remove(sample_path)

    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    tok.save(tok_path)
    print(f"Tokenizer sauvegardé : {tok_path} (vocab_size={tok.vocab_size})")

    # ------------------------------------------------------------------
    # 2) Tokenisation en streaming du corpus ENTIER -> train.bin / val.bin
    #    Le fichier source n'est jamais chargé en entier : on écrit
    #    directement sur disque au fur et à mesure de la lecture.
    # ------------------------------------------------------------------
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    n_train_tokens = 0
    n_val_tokens = 0

    print("Tokenisation en streaming du corpus complet...")
    with open(train_path, "wb") as ftrain, open(val_path, "wb") as fval:
        for i, chunk in enumerate(iter_chunks(args.input)):
            ids = np.asarray(tok.encode(chunk), dtype=np.uint16)
            # split train/val À L'INTÉRIEUR du bloc (pas en alternant des blocs
            # entiers) : chaque bloc contribue aux deux ensembles, donc la
            # validation reste bien répartie sur tout le corpus, même s'il
            # tient en un seul bloc (petit fichier).
            split_idx = int(len(ids) * (1 - args.val_frac))
            train_ids, val_ids = ids[:split_idx], ids[split_idx:]
            train_ids.tofile(ftrain)
            val_ids.tofile(fval)
            n_train_tokens += len(train_ids)
            n_val_tokens += len(val_ids)
            if i % 50 == 0:
                print(f"  bloc {i:5d} -- {n_train_tokens:,} tokens train / {n_val_tokens:,} tokens val")

    meta = {
        "vocab_size": tok.vocab_size,
        "n_train_tokens": n_train_tokens,
        "n_val_tokens": n_val_tokens,
        "source_file": os.path.abspath(args.input),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = n_train_tokens + n_val_tokens
    print(f"\nTerminé : {total:,} tokens ({n_train_tokens:,} train / {n_val_tokens:,} val)")
    print(f"Fichiers écrits dans : {args.out_dir}/ (train.bin, val.bin, tokenizer.json, meta.json)")
    print(f"Lance ensuite : python train.py --data_dir {args.out_dir} --steps 5000")


if __name__ == "__main__":
    main()