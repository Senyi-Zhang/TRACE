# -*- coding: utf-8 -*-
# train_stage2_joint_v3_partial.py
# DeBERTa-v3-base + 语义树 + 联合训练（仅解冻最后 N 层），适配 8GB 显存

import os, re, math, json, random, argparse
from typing import List, Dict, Any, Tuple
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from wordfreq import zipf_frequency

# 你的实现文件
from NodeEncoder import NodeEncoder              # 需要 enc/tok + proj_self/proj_child/gate/norm/dropout
from semantic_tree import claim_to_span_tree     # 你的 CSPT

# --------- Utils ----------
def seed_all(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

LABEL_TRUE  = {"1","true","entailed","entails","support","supports","supported","yes","pos","positive","SUPPORTED"}
LABEL_FALSE = {"0","false","refuted","contradiction","contradicts","no","neg","negative","REFUTED"}
def norm_label(x):
    if isinstance(x,(int,float)): return int(round(float(x)))
    s=str(x).strip().lower()
    if s.isdigit(): return int(s)
    if s in {w.lower() for w in LABEL_TRUE}:  return 1
    if s in {w.lower() for w in LABEL_FALSE}: return 0
    raise ValueError(f"Unrecognized label: {x}")

@lru_cache(maxsize=100000)
def cached_tree(text: str) -> Dict[str,Any]:
    return claim_to_span_tree(text)

def is_number(s: str) -> bool:
    try: float(s); return True
    except: return False

# --------- Dataset ----------
class TabFactTreeJsonl(Dataset):
    """
    支持两种格式之一：
    A) {claim/statement, table:{header, rows, caption}, label}
    B) {claim/statement, header:[...], rows:[[...], ...], caption:str, label}
    """
    def __init__(self, path: str, max_samples: int=None):
        self.recs = []
        ok = skip = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                claim = r.get("statement") or r.get("claim")
                label = r.get("label")
                table = r.get("table")
                if table is None:
                    header  = r.get("header")
                    rows    = r.get("rows")
                    caption = r.get("caption")
                    if header is not None or rows is not None or caption is not None:
                        header  = [] if header is None else list(header)
                        caption = "" if caption is None else str(caption)
                        if isinstance(rows, list):
                            rows = [[str(x) for x in row] for row in rows]
                        else:
                            rows = []
                        table = {"header": header, "rows": rows, "caption": caption}
                if claim is None or label is None or table is None:
                    skip += 1; continue
                try:
                    y = norm_label(label)
                except:
                    skip += 1; continue
                self.recs.append((str(claim), table, y)); ok += 1
                if max_samples and len(self.recs) >= max_samples: break
        assert len(self.recs)>0, f"No usable samples in {path}. ok={ok}, skip={skip}"
    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]

def collate(batch):
    claims, tables, labels = zip(*batch)
    return list(claims), list(tables), torch.tensor(labels, dtype=torch.long)

# --------- Row linearization & rerank（与 Stage-1 行串一致） ----------
def linearize_row(table: Dict[str,Any]) -> List[str]:
    out = []
    cap = str(table.get("caption","") or "")
    header = list(map(str, table.get("header", [])))
    for r in table.get("rows", []):
        s = f"[TITLE]  {cap} |  [ROW] "
        for i in range(min(len(header), len(r))):
            s += f"{header[i]}: {str(r[i])}   "
        out.append(s)
    return out

def rerank(query: str, table: Dict[str,Any], k: int=6) -> List[str]:
    q = re.sub(r"[^\w\s]", " ", str(query))
    toks = q.split()
    rarity = {w: (8 if is_number(w) else 8 - zipf_frequency(w, "en")) for w in toks}
    cand = linearize_row(table)
    scored = []
    for row in cand:
        s = 0.0
        for w in toks:
            if w in row: s += rarity[w]
        scored.append((s, row))
    scored.sort(key=lambda x: x[1], reverse=True)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

# --------- Model ----------
class Stage2JointV3Partial(nn.Module):
    """
    DeBERTa-v3-base 与门控组件联合训练，但只解冻 encoder 的最后 N 层。
    - 每样本：收集整棵树的 (node_text, ctx_for_node)，按 node_micro_batch 过 encoder（保梯度）
    - 孩子自查询注意力 + 门控融合
    - 根向量 -> 线性头
    """
    def __init__(self, node: NodeEncoder, num_labels=2, max_len=160,
                 K_self=6, joiner="  ||  ", child_method="attn",
                 node_mb_size: int = 8, grad_ckpt: bool = True):
        super().__init__()
        self.node = node
        self.max_len = max_len
        self.K_self = K_self
        self.joiner = joiner
        self.child_method = child_method
        self.node_mb_size = node_mb_size

        h = self.node.enc.config.hidden_size
        self.head = nn.Linear(h, num_labels)

        if grad_ckpt and hasattr(self.node.enc, "gradient_checkpointing_enable"):
            try:
                self.node.enc.gradient_checkpointing_enable()
                if hasattr(self.node.enc.config, "use_cache"):
                    self.node.enc.config.use_cache = False
            except Exception:
                pass

    def _collect_pairs(self, root: Dict[str,Any], table: Dict[str,Any]) -> Tuple[List[Tuple[int,str,str]], Dict[int,List[int]]]:
        pairs = []  # [(nid, text, ctx)]
        edges = {}  # nid -> [child_ids]
        stack = [(0, root)]
        next_id = 1
        id_map = {id(root): 0}
        while stack:
            nid, nd = stack.pop()
            rows = rerank(nd["text"], table, k=self.K_self)
            ctx = self.joiner.join(rows)
            pairs.append((nid, nd["text"], ctx))
            ch_ids = []
            for ch in nd.get("children", []):
                cid = id_map.get(id(ch))
                if cid is None:
                    cid = next_id; next_id += 1
                    id_map[id(ch)] = cid
                ch_ids.append(cid)
                stack.append((cid, ch))
            edges[nid] = ch_ids
        return pairs, edges

    @staticmethod
    def _masked_mean(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        s = (hs*m).sum(1)
        d = m.sum(1).clamp_min(1e-6)
        return s / d

    def _encode_pairs_with_grad(self, pairs: List[Tuple[int,str,str]]) -> Dict[int, torch.Tensor]:
        device = next(self.parameters()).device
        ids = [p[0] for p in pairs]
        texts_a = [p[1] for p in pairs]
        texts_b = [p[2] for p in pairs]

        vecs = []
        B = len(pairs)
        mb = max(1, self.node_mb_size)
        for st in range(0, B, mb):
            ed = min(B, st+mb)
            tok = self.node.tok(texts_a[st:ed], texts_b[st:ed],
                                truncation=True, max_length=self.max_len,
                                padding=True, return_tensors="pt").to(device)
            out = self.node.enc(**tok).last_hidden_state          # [mb,L,H]
            pooled = self._masked_mean(out, tok["attention_mask"])# [mb,H]
            vecs.append(F.normalize(pooled, p=2, dim=-1))
        vec = torch.cat(vecs, dim=0)                               # [B,H]
        return {i: v for i, v in zip(ids, vec)}

    def _compose_from_leaves(self, nid: int, edges: Dict[int,List[int]], self_vecs: Dict[int,torch.Tensor]) -> torch.Tensor:
        self_emb = self_vecs[nid]               # [H]
        child_ids = edges.get(nid, [])
        if not child_ids:
            return self_emb
        child_list = [self._compose_from_leaves(cid, edges, self_vecs) for cid in child_ids]
        childs = torch.stack(child_list, 0)     # [K,H]

        if self.child_method == "attn":
            q = F.normalize(self_emb, p=2, dim=-1)
            k = F.normalize(childs, p=2, dim=-1)
            w = torch.softmax(k @ q, dim=0).unsqueeze(-1)
            ctx = (childs * w).sum(0)
        else:
            ctx = childs.mean(0)

        h_self  = self.node.proj_self(self_emb)
        h_child = self.node.proj_child(ctx)
        g = torch.sigmoid(self.node.gate(torch.cat([h_self, h_child], dim=-1)))
        fused = g * h_child + (1 - g) * h_self
        out = self.node.norm(self_emb + self.node.dropout(fused))
        return out

    def forward(self, claims: List[str], tables: List[Dict[str,Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        roots = [cached_tree(c) for c in claims]
        Hs = []
        for c_root, table in zip(roots, tables):
            pairs, edges = self._collect_pairs(c_root, table)
            self_vecs = self._encode_pairs_with_grad(pairs)      # {nid: [H]}
            h_root = self._compose_from_leaves(0, edges, self_vecs)
            Hs.append(h_root.to(device))
        H = torch.stack(Hs, 0)
        return self.head(H)

# --------- 只解冻最后 N 层的辅助 ----------
def freeze_all(m: nn.Module):
    for p in m.parameters(): p.requires_grad = False

def unfreeze_last_n_layers_for_deberta(enc: nn.Module, n: int):
    """
    适配 DeBERTaV3 (HF: DebertaV2Model) 结构：enc.encoder.layer 是列表
    只解冻最后 n 层；可选解冻 encoder.LayerNorm
    """
    try:
        layers = enc.encoder.layer
    except AttributeError as e:
        raise RuntimeError(f"Unexpected encoder structure: {type(enc)}") from e

    L = len(layers)
    n = max(0, min(n, L))
    for i, block in enumerate(layers):
        req = (i >= L - n)
        for p in block.parameters():
            p.requires_grad = req

    # （可选）解冻顶层 LayerNorm
    if hasattr(enc.encoder, "LayerNorm"):
        for p in enc.encoder.LayerNorm.parameters(): p.requires_grad = True

# --------- Eval ----------
@torch.no_grad()
def evaluate(model: Stage2JointV3Partial, dl, device):
    model.eval()
    total = correct = 0
    for claims, tables, y in tqdm(dl, desc="Validate", leave=False, dynamic_ncols=True):
        y = y.to(device)
        logits = model(claims, tables)
        pred = logits.argmax(-1)
        correct += (pred==y).sum().item()
        total   += y.numel()
    return correct / max(1,total)

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train", default="D:\\my_datasets\\tabfact\\train.jsonl")
    ap.add_argument("--dev",   default="D:\\my_datasets\\tabfact\\validation.jsonl")
    ap.add_argument("--test",  default="D:\\my_datasets\\tabfact\\test.jsonl")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    # 模型
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--K_self", type=int, default=6)          # 小点更省显存
    ap.add_argument("--max_len", type=int, default=160)       # 8GB 友好
    ap.add_argument("--child_method", choices=["attn","mean"], default="attn")
    ap.add_argument("--node_mb_size", type=int, default=8)    # 每样本内节点 micro-batch
    ap.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--last_n_unfrozen", type=int, default=6) # 只解冻最后 N 层
    # 优化
    ap.add_argument("--batch_size", type=int, default=1)      # 外层 batch=1
    ap.add_argument("--grad_accum", type=int, default=16)     # 梯度累积撑等效 batch
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr_backbone", type=float, default=8e-6)
    ap.add_argument("--lr_head",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    # 运行
    ap.add_argument("--num_workers", type=int, default=0)     # Windows 建议 0
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="stage2_joint_v3_partial.pt")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NodeEncoder：加载 DeBERTa-v3-base
    node = NodeEncoder(model_name=args.model_name, device=device)
    node.enc.train()  # 覆盖 NodeEncoder.__init__ 里的 .eval()

    # 门控偏置小技巧：初始更偏自分支
    with torch.no_grad():
        if getattr(node, "gate", None) is not None and getattr(node.gate, "bias", None) is not None:
            node.gate.bias.fill_(-1.0)

    model = Stage2JointV3Partial(
        node, num_labels=2, max_len=args.max_len, K_self=args.K_self,
        child_method=args.child_method, node_mb_size=args.node_mb_size,
        grad_ckpt=args.grad_ckpt
    ).to(device)

    # 只解冻最后 N 层
    freeze_all(model.node.enc)
    unfreeze_last_n_layers_for_deberta(model.node.enc, args.last_n_unfrozen)
    # 门控/投影/Norm/头始终参与训练
    for m in [model.node.proj_self, model.node.proj_child, model.node.gate, model.node.norm, model.head]:
        for p in m.parameters(): p.requires_grad = True

    # 数据
    train_full = TabFactTreeJsonl(args.train, max_samples=args.max_train_samples)
    if args.dev:
        dev_set = TabFactTreeJsonl(args.dev)
        train_set = train_full
    else:
        recs = train_full.recs[:]; random.shuffle(recs)
        n_val = max(1, int(len(recs)*args.val_ratio))
        dev_recs = recs[:n_val]; train_recs = recs[n_val:]
        class ListDS(Dataset):
            def __init__(self, recs): self.recs=recs
            def __len__(self): return len(self.recs)
            def __getitem__(self,i): return self.recs[i]
        train_set = ListDS(train_recs); dev_set = ListDS(dev_recs)
    test_set = TabFactTreeJsonl(args.test) if args.test else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                              collate_fn=collate)
    dev_loader   = DataLoader(dev_set,   batch_size=max(8,args.batch_size), shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                              collate_fn=collate)
    test_loader  = (DataLoader(test_set, batch_size=max(8,args.batch_size), shuffle=False,
                               num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                               collate_fn=collate) if test_set else None)

    # 参数组：encoder（最后 N 层） vs 门控/头
    enc_params = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("node.enc.")]
    gate_params= [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("node.enc.")]
    optim = torch.optim.AdamW(
        [{"params": enc_params,  "lr": args.lr_backbone},
         {"params": gate_params, "lr": args.lr_head}],
        weight_decay=args.weight_decay
    )

    # 线性 warmup + 线性衰减
    steps_per_epoch = math.ceil(len(train_loader) / max(1,args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    def lr_lambda(step):
        wrm = int(total_steps * args.warmup_ratio)
        if step < wrm:
            return step / max(1,wrm)
        else:
            remain = step - wrm
            return max(0.0, 1.0 - remain / max(1,total_steps-wrm))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type=="cuda"))

    print(f"[info] train={len(train_set)}, dev={len(dev_set)}, "
          f"batch={args.batch_size}, accum={args.grad_accum}, "
          f"K_self={args.K_self}, max_len={args.max_len}, node_mb={args.node_mb_size}, "
          f"last_n_unfrozen={args.last_n_unfrozen}")

    best_dev, best_state = 0.0, None
    global_updates = 0
    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                    desc=f"Epoch {ep}/{args.epochs}", dynamic_ncols=True)
        optim.zero_grad(set_to_none=True)

        for step, (claims, tables, y) in pbar:
            y = y.to(device)
            with torch.amp.autocast("cuda", enabled=(args.fp16 and device.type=="cuda")):
                logits = model(claims, tables)
                loss = F.cross_entropy(logits, y)

            loss = loss / max(1, args.grad_accum)     # 累积平均
            running += loss.item() * max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                sched.step()
                global_updates += 1
                optim.zero_grad(set_to_none=True)

            lr_e = optim.param_groups[0]["lr"]; lr_g = optim.param_groups[1]["lr"]
            pbar.set_postfix(loss=f"{running/max(1,step):.4f}", lr_e=f"{lr_e:.2e}", lr_g=f"{lr_g:.2e}",
                             updates=global_updates)

        dev_acc = evaluate(model, dev_loader, device)
        print(f"Epoch {ep} | dev acc {dev_acc:.4f}")
        if dev_acc > best_dev + 1e-4:
            best_dev = dev_acc
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args),
                        "model_name": args.model_name}, args.ckpt)
            print("  (saved best)")

    # 测试
    if best_state is None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(best_state)

    if test_loader is not None:
        test_acc = evaluate(model, test_loader, device)
        print(f"Best dev acc {best_dev:.4f} | test acc {test_acc:.4f}")
    else:
        print(f"Best dev acc {best_dev:.4f} | (no test set provided)")

if __name__ == "__main__":
    main()
