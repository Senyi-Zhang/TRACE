import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Union, Sequence


class PMA(nn.Module):
    """
    Pooling by Multihead Attention
    输入:
        X: [K, D] 或 [B, K, D]
    输出:
        pooled: [D] / [B, D]   （默认 1 个 seed）
    """
    def __init__(self, d_model: int, num_heads: int = 8, num_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_seeds = num_seeds
        self.seed = nn.Parameter(torch.randn(num_seeds, d_model))
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if X.dim() == 2:
            X = X.unsqueeze(0)   # [1, K, D]
            squeeze_back = True

        B = X.size(0)
        S = self.seed.unsqueeze(0).expand(B, -1, -1)   # [B, num_seeds, D]

        attn_out, _ = self.mha(S, X, X, need_weights=False)   # [B, num_seeds, D]
        H = self.ln1(S + attn_out)
        H = self.ln2(H + self.ffn(H))

        if self.num_seeds == 1:
            H = H[:, 0, :]   # [B, D]

        if squeeze_back:
            H = H.squeeze(0)

        return H


class FiLMFusion(nn.Module):
    """
    用 child_ctx 调制 self_emb:
        gamma, beta = MLP(child_ctx)
        mod = gamma * W_self(self_emb) + beta
        out = LN(self_emb + Dropout(mod + W_child(child_ctx)))
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj_self = nn.Linear(d_model, d_model)
        self.proj_child = nn.Linear(d_model, d_model)

        self.to_gamma_beta = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, self_emb: torch.Tensor, child_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if child_ctx is None:
            return self_emb

        h_self = self.proj_self(self_emb)       # [D]
        h_child = self.proj_child(child_ctx)    # [D]

        gamma, beta = self.to_gamma_beta(child_ctx).chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)   # 比直接用 gamma 更稳，初始更接近 identity

        mod = gamma * h_self + beta       # FiLM
        out = self.norm(self_emb + self.dropout(mod + h_child))
        out = F.normalize(out, p=2, dim=-1)
        return out


class NodeEncoderPMAFiLM(nn.Module):
    """
    改成 PMA + FiLM 的版本：

    1. 叶子:
       self_emb = encode_text(query)

    2. 非叶:
       - 如果 row_text 是 str: 视作 1 条 evidence
       - 如果 row_text 是 list[str]: 每条 (query, row) 编码成 row embedding
       - 再用 PMA(rows) 得到 self_emb

    3. children:
       child_ctx = PMA(child_list)

    4. fuse:
       out = FiLM(self_emb | child_ctx)
    """
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device=None,
        dropout: float = 0.1,
        num_heads: int = 8
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device)
        print("Model loaded.")

        dim = self.enc.config.hidden_size

        # 一个给 rows 聚合，一个给 children 聚合
        self.row_pma = PMA(d_model=dim, num_heads=num_heads, num_seeds=1, dropout=dropout)
        self.child_pma = PMA(d_model=dim, num_heads=num_heads, num_seeds=1, dropout=dropout)

        self.film_fuse = FiLMFusion(d_model=dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.to(self.device)

    @staticmethod
    def _masked_mean(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hs: [B,L,H], mask: [B,L] or [B,L,1]
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.float()
        s = (hs * mask).sum(1)
        d = mask.sum(1).clamp_min(1e-6)
        return s / d

    def encode_text(self, text: str, max_len: int = 256) -> torch.Tensor:
        """
        单文本 -> [D]
        """
        batch = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(self.device)

        h = self.enc(**batch).last_hidden_state          # [1, L, D]
        vec = self._masked_mean(h, batch.attention_mask) # [1, D]
        vec = F.normalize(vec.squeeze(0), p=2, dim=-1)   # [D]
        return vec

    def encode_pairs(self, query: str, rows: Sequence[str], max_len: int = 256) -> torch.Tensor:
        """
        批量编码多个 (query, row) -> [K, D]
        """
        if len(rows) == 0:
            rows = [""]

        queries = [query] * len(rows)
        batch = self.tok(
            queries,
            list(rows),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(self.device)

        h = self.enc(**batch).last_hidden_state              # [K, L, D]
        vecs = self._masked_mean(h, batch.attention_mask)    # [K, D]
        vecs = F.normalize(vecs, p=2, dim=-1)
        return vecs

    def encode_node_self(
        self,
        query: str,
        row_text: Optional[Union[str, List[str]]],
        is_leaf: bool,
        max_len: int = 256
    ) -> torch.Tensor:
        """
        当前节点自己的表示 self_emb
        """
        if is_leaf:
            self_emb = self.encode_text(query, max_len=max_len)
            return self_emb

        # 非叶：允许一条或多条 evidence rows
        if row_text is None:
            rows = [""]
        elif isinstance(row_text, str):
            rows = [row_text]
        else:
            rows = list(row_text)

        row_embs = self.encode_pairs(query, rows, max_len=max_len)  # [K, D]

        if row_embs.size(0) == 1:
            self_emb = row_embs[0]
        else:
            self_emb = self.row_pma(row_embs)   # [D]

        self_emb = F.normalize(self_emb, p=2, dim=-1)
        return self_emb

    def summarize_children(self, child_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        child_list -> PMA -> child_ctx
        """
        if not child_list:
            return None

        childs = torch.stack([c.to(self.device) for c in child_list], dim=0)  # [K, D]

        if childs.size(0) == 1:
            ctx = childs[0]
        else:
            ctx = self.child_pma(childs)   # [D]

        ctx = F.normalize(ctx, p=2, dim=-1)
        return ctx

    def forward_once(
        self,
        query: str,
        row_text: Optional[Union[str, List[str]]],
        child_list: List[torch.Tensor],
        is_leaf: bool,
        max_len: int = 256,
        return_parts: bool = False
    ):
        """
        当前节点前向:
            self_emb = 当前节点自己的表示
            child_ctx = PMA(children)
            out = FiLM(self_emb | child_ctx)
        """
        self_emb = self.encode_node_self(
            query=query,
            row_text=row_text,
            is_leaf=is_leaf,
            max_len=max_len
        )  # [D]

        child_ctx = self.summarize_children(child_list)  # [D] or None
        out = self.film_fuse(self_emb, child_ctx)        # [D]

        if return_parts:
            return out, {
                "self_emb": self_emb,
                "child_ctx": child_ctx
            }
        return out