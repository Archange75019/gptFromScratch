"""
Mini-LLaMA (Projet 2) : Transformer décodeur moderne codé à la main.

Par rapport au Projet 1 (GPT-2 like), trois briques changent :

    LayerNorm             ->  RMSNorm   (pas de centrage, pas de biais : moins cher, aussi stable)
    Embedding de position ->  RoPE      (rotation de Q et K selon la position, pas d'embedding appris)
    MLP GELU (x4)         ->  SwiGLU    (MLP à porte : silu(x W_gate) * (x W_up), puis W_down)

Le reste est identique : pré-normalisation, connexions résiduelles, attention
causale codée à la main (on garde la version "visible" pour les heatmaps).

Autres habitudes LLaMA reprises : aucun biais dans les couches linéaires,
dropout à 0 par défaut, init des projections résiduelles réduite avec la profondeur.

Bloc :
    x -> RMSNorm -> Self-Attention causale (+RoPE sur Q,K) -> +résiduel
      -> RMSNorm -> SwiGLU                                   -> +résiduel
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
    dropout: float = 0.0       # LLaMA n'utilise pas de dropout ; 0.0-0.1 selon la taille du corpus
    rope_theta: float = 10000.0   # base des fréquences RoPE
    norm_eps: float = 1e-5        # epsilon de RMSNorm
    ffn_multiple_of: int = 64     # la dimension cachée de SwiGLU est arrondie à ce multiple
    tie_embeddings: bool = False  # True : lm_head partage ses poids avec l'embedding de tokens
    flash_attn: bool = False      # True : F.scaled_dot_product_attention (plus rapide, même résultat)


# ----------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """x / rms(x) * poids. Comme LayerNorm sans soustraction de la moyenne ni biais."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()  # le calcul de la norme se fait en float32, même en bf16
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_f.type_as(x) * self.weight


# ----------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ----------------------------------------------------------------------
def precompute_rope(head_dim: int, block_size: int, theta: float = 10000.0):
    """Tables cos/sin de forme (block_size, head_dim/2).

    Chaque paire de dimensions (i, i + head_dim/2) est tournée d'un angle
    position * theta^(-2i/head_dim) : les dimensions "basses" tournent vite
    (info locale), les "hautes" lentement (info à longue portée).
    """
    assert head_dim % 2 == 0, "RoPE exige un head_dim pair (n_embd / n_head)"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(block_size).float()
    angles = torch.outer(positions, inv_freq)  # (block_size, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x : (B, n_head, T, head_dim) ; cos/sin : (T, head_dim/2). Rotation par paires."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.type_as(x)


# ----------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Self-attention multi-têtes causale, avec RoPE appliqué à Q et K (pas à V)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE exige un head_dim pair"
        self.flash = config.flash_attn
        self.dropout_p = config.dropout

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def _qkv(self, x, cos, sin):
        """Projette x en Q, K, V (B, nh, T, hd) et applique RoPE à Q et K."""
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        return q, k, v

    def attention_weights(self, x, cos, sin):
        """Poids d'attention softmax (B, nh, T, T) : utilisé par visualize_attention.py."""
        T = x.size(1)
        q, k, _ = self._qkv(x, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        return F.softmax(scores, dim=-1)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self._qkv(x, cos, sin)

        if self.flash:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


# ----------------------------------------------------------------------
# SwiGLU
# ----------------------------------------------------------------------
class FeedForward(nn.Module):
    """SwiGLU : down( silu(gate(x)) * up(x) ).

    Trois matrices au lieu de deux : pour garder le même nombre de paramètres
    que le MLP x4 classique, la dimension cachée vaut ~ 8/3 * n_embd (arrondie
    au multiple de `ffn_multiple_of` supérieur) au lieu de 4 * n_embd.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = config.ffn_multiple_of * math.ceil(hidden / config.ffn_multiple_of)
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """Un bloc Transformer = attention + SwiGLU, chacun avec pré-RMSNorm et résiduel."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # plus de position_embedding : la position est injectée par RoPE dans chaque attention
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # tables RoPE : dérivées de la config, donc non sauvegardées dans le checkpoint
        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # init "résiduelle" (GPT-2 / LLaMA) : les projections qui écrivent dans le flux
        # résiduel sont réduites en 1/sqrt(2*n_layer) pour que la variance ne gonfle pas
        # avec la profondeur (2 ajouts résiduels par bloc)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MiniGPT (LLaMA-style) initialisé : {n_params/1e6:.2f}M paramètres")

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

        x = self.dropout(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Génère du texte token par token (sampling autoregressif)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train(was_training)
        return idx