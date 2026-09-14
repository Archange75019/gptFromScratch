"""
Mini-GPT : Transformer décodeur codé à la main.

Inspiré de nanoGPT (Andrej Karpathy), réécrit avec des noms explicites et
des commentaires pédagogiques pour un article LinkedIn / une démo.

Architecture d'un bloc Transformer décodeur :

    x -> LayerNorm -> Self-Attention (causale) -> +résiduel
      -> LayerNorm -> FeedForward (MLP)         -> +résiduel

On empile `n_layer` de ces blocs, précédés d'un embedding de tokens +
embedding de position, et suivis d'une projection finale vers le vocabulaire.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256      # longueur max de contexte (fenêtre d'attention)
    n_layer: int = 4           # nombre de blocs Transformer empilés
    n_head: int = 4            # nombre de têtes d'attention
    n_embd: int = 128          # dimension des embeddings
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Self-attention multi-têtes, codée à la main (pas de nn.MultiheadAttention)
    pour bien voir les 3 étapes : Q/K/V -> scores -> pondération des valeurs.

    "Causale" = chaque position ne peut regarder que les positions
    précédentes (masque triangulaire), sinon le modèle "tricherait" en
    regardant le futur pendant l'entraînement.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # une seule matrice projette x vers Q, K et V en même temps (plus efficace)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # masque triangulaire inférieur : position i ne voit que 0..i
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # on répartit C en (n_head, head_dim) pour paralléliser les têtes
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scores d'attention = similarité Q.K^T, normalisée par sqrt(head_dim)
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        # on masque le futur en mettant -inf avant le softmax
        att_scores = att_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v  # (B, nh, T, hd) -- moyenne pondérée des valeurs
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # on recolle les têtes

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """MLP position-wise : chaque token est traité indépendamment, expansion x4."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Un bloc Transformer = attention + MLP, chacun avec pré-normalisation et résiduel."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # connexion résiduelle autour de l'attention
        x = x + self.mlp(self.ln2(x))   # connexion résiduelle autour du MLP
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MiniGPT initialisé : {n_params/1e6:.2f}M paramètres")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "séquence plus longue que block_size"

        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Génère du texte token par token (sampling autoregressif)."""
        was_training = self.training  # on restaurera cet état à la fin
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train(was_training)  # remet le modèle dans l'état où il était avant l'appel
        return idx