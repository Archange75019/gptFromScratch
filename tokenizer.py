"""
Tokenizer BPE (Byte-Pair Encoding), entraîné avec la librairie HuggingFace
`tokenizers` (pip install tokenizers).

Vocabulaire de quelques milliers de tokens, chaque token couvrant en
moyenne 3-4 caractères -- indispensable pour un corpus de plusieurs
centaines de Mo / quelques Go comme Wikipédia, utilisé avec
prepare_data.py (tokenisation en streaming vers train.bin/val.bin, jamais
chargé entier en RAM).

Byte-level : le pré-tokenizer travaille sur les octets UTF-8 bruts, pas sur
des caractères Unicode -- ça garantit qu'AUCUN texte ne peut jamais
produire un caractère "inconnu" (contrairement à un tokenizer
character-level classique), ce qui est important sur un corpus web où on
croise toutes sortes de caractères exotiques, emojis, fautes d'encodage,
etc.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class BPETokenizer:
    """Wrapper léger autour de `tokenizers.Tokenizer` (BPE byte-level)."""

    def __init__(self, hf_tokenizer: Tokenizer):
        self._tok = hf_tokenizer
        self.vocab_size = self._tok.get_vocab_size()

    @classmethod
    def train(cls, file_paths: list[str], vocab_size: int = 8000, min_frequency: int = 2):
        """Entraîne un nouveau tokenizer BPE sur un (ou plusieurs) fichier(s) texte.

        vocab_size doit rester < 65536 : les IDs de tokens sont stockés en
        uint16 sur disque par prepare_data.py pour économiser la mémoire
        (8x moins que des int64) -- 8000 à 32000 est un choix courant pour
        un corpus mono-langue de cette échelle (à comparer aux ~50k de GPT-2
        ou aux ~100k de tokenizers multilingues modernes).
        """
        assert vocab_size < 65536, "vocab_size doit rester < 65536 (contrainte du stockage uint16)"

        hf_tok = Tokenizer(models.BPE(unk_token=None))
        hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        hf_tok.train(files=file_paths, trainer=trainer)
        return cls(hf_tok)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode plusieurs textes d'un coup (plus rapide qu'un encode() par boucle)."""
        return [e.ids for e in self._tok.encode_batch(texts)]

    def decode(self, indices: list[int]) -> str:
        return self._tok.decode(indices)

    def token_str(self, token_id: int) -> str:
        """Renvoie la représentation texte d'un seul token (utile pour les
        visualisations, ex: labels d'une heatmap d'attention)."""
        return self._tok.decode([token_id])

    def save(self, path: str):
        self._tok.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(Tokenizer.from_file(path))


if __name__ == "__main__":
    # petit test manuel
    import tempfile
    import os

    text_fr = ("Le renard brun rapide saute par-dessus le chien paresseux. " * 200
               + "Les modèles de langage s'entraînent sur de grands corpus de texte. " * 200)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text_fr)
        tmp_path = f.name

    tok = BPETokenizer.train([tmp_path], vocab_size=300)
    ids = tok.encode("Le renard rapide.")
    print("vocab_size:", tok.vocab_size)
    print("encoded:", ids)
    print("decoded:", tok.decode(ids))
    assert tok.decode(ids) == "Le renard rapide."
    os.unlink(tmp_path)
    print("OK")