"""
Télécharge un corpus Wikipédia français EN STREAMING, prêt à être
tokenisé par prepare_data.py puis utilisé par train.py.

A LANCER SUR TA MACHINE (pas dans un sandbox restreint) car il a besoin
d'un accès internet normal.

Nécessite : pip install datasets

Usage :
    python download_corpus.py --max_mb 500
    python download_corpus.py --max_mb 2000 --out corpus_fr_large.txt

Le dataset Wikipédia FR fait plusieurs dizaines de Go sur Hugging Face --
on ne le télécharge JAMAIS en entier : `streaming=True` ouvre un flux de
documents qu'on écrit sur disque au fil de l'eau, en s'arrêtant dès que
--max_mb est atteint. Rien n'est chargé entièrement en RAM, ni le dataset
distant, ni le fichier de sortie.
"""

import argparse

WIKIPEDIA_SOURCE = dict(path="wikimedia/wikipedia", name="20231101.fr", split="train")


def download_streaming_corpus(out_path: str, max_mb: int):
    """
    Télécharge le corpus Wikipédia FR en streaming.

    Hugging Face ne rapatrie les articles qu'au fur et à mesure qu'on les
    consomme, jamais le dataset complet. On écrit chaque article dans
    `out_path` dès qu'on le reçoit, et on s'arrête dès que `max_mb` Mo ont
    été écrits. À aucun moment le corpus n'est gardé entier en mémoire ou
    sur disque au-delà de ce qui est demandé.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Ce script nécessite la librairie `datasets` : pip install datasets"
        )

    print(f"Connexion au dataset '{WIKIPEDIA_SOURCE['path']}' "
          f"({WIKIPEDIA_SOURCE['name']}) en streaming...")
    ds = load_dataset(
        WIKIPEDIA_SOURCE["path"],
        WIKIPEDIA_SOURCE["name"],
        split=WIKIPEDIA_SOURCE["split"],
        streaming=True,
    )

    max_bytes = max_mb * 1_000_000
    written = 0
    n_docs = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            written += len(text.encode("utf-8"))
            n_docs += 1
            if n_docs % 500 == 0:
                print(f"  {n_docs:,} articles -- {written/1e6:.1f} Mo écrits")
            if written >= max_bytes:
                break

    print(f"Terminé : {out_path} ({written/1e6:.1f} Mo, {n_docs:,} articles)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_mb", type=int, default=500,
                        help="Taille cible en Mo (défaut : 500 Mo).")
    parser.add_argument("--out", type=str, default="corpus_wikipedia_fr.txt")
    args = parser.parse_args()

    download_streaming_corpus(args.out, args.max_mb)
    print(f"Lance ensuite : python prepare_data.py --input {args.out} --out_dir data_prepared")


if __name__ == "__main__":
    main()