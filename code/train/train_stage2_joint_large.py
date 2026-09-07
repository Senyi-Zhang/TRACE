import os, re, math, json, random, argparse
from typing import List, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from wordfreq import zipf_frequency
from functools import lru_cache

# ---- 你的实现 ----
from NodeEncoder import NodeEncoder              # 使用其中的 enc/tok + proj_self/proj_child/gate/norm/dropout
from semantic_tree import claim_to_span_tree     # 你的 CSPT

# --------------- Utils ---------------
def seed_all(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

# --------------- Dataset ---------------
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

# --------------- Row linearization & rerank（与 Stage-1 行串格式对齐） ---------------
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

def rerank(query: str, table: Dict[str,Any], k: int=8) -> List[str]:
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

# --------------- Model ---------------
class Stage2JointLarge(nn.Module):
    """
    DeBERTa-v3-large 与门控组件**联合训练**。
    - 每样本：先收集整棵树的 (node_text, ctx_for_node)，用 micro-batch 一次性过 encoder（保梯度）
    - 孩子汇总（self-queried attention）+ 门控融合（proj_self/proj_child/gate/norm）
    - 根向量 -> 线性头二分类
    """
    def __init__(self, node: NodeEncoder, num_labels=2, max_len=192,
                 K_self=8, joiner="  ||  ", child_method="attn",
                 node_mb_size: int = 64, grad_ckpt: bool = True):
        super().__init__()
        self.node = node
        self.max_len = max_len
        self.K_self = K_self
        self.joiner = joiner
        self.child_method = child_method
        self.node_mb_size = node_mb_size

        h = self.node.enc.config.hidden_size
        self.head = nn.Linear(h, num_labels)

        # 开启 gradient checkpointing（可显著省显存）
        if grad_ckpt and hasattr(self.node.enc, "gradient_checkpointing_enable"):
            try:
                self.node.enc.gradient_checkpointing_enable()
                if hasattr(self.node.enc.config, "use_cache"):
                    self.node.enc.config.use_cache = False
            except Exception:
                pass  # 个别版本可能不支持

    # --- 收集整棵树需要编码的 pairs ---
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
        # hs:[B,L,H], mask:[B,L]
        m = mask.unsqueeze(-1).float()
        s = (hs*m).sum(1)
        d = m.sum(1).clamp_min(1e-6)
        return s / d

    # --- 一次性（按 micro-batch）编码所有节点对，保梯度 ---
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

    # --- 自底向上融合（注意力 + 门控） ---
    def _compose_from_leaves(self, nid: int, edges: Dict[int,List[int]], self_vecs: Dict[int,torch.Tensor]) -> torch.Tensor:
        self_emb = self_vecs[nid]               # [H] with grad
        child_ids = edges.get(nid, [])
        if not child_ids:
            return self_emb
        child_list = [self._compose_from_leaves(cid, edges, self_vecs) for cid in child_ids]
        childs = torch.stack(child_list, 0)     # [K,H]

        if self.child_method == "attn":
            q = F.normalize(self_emb, p=2, dim=-1)  # [H]
            k = F.normalize(childs, p=2, dim=-1)    # [K,H]
            w = torch.softmax(k @ q, dim=0).unsqueeze(-1)  # [K,1]
            ctx = (childs * w).sum(0)
        else:
            ctx = childs.mean(0)

        h_self  = self.node.proj_self(self_emb)
        h_child = self.node.proj_child(ctx)
        g = torch.sigmoid(self.node.gate(torch.cat([h_self, h_child], dim=-1)))  # [1]
        fused = g * h_child + (1 - g) * h_self
        out = self.node.norm(self_emb + self.node.dropout(fused))
        return out

    # --- 前向：每样本只跑一次 encoder（内部用 micro-batch），其余全保梯度 ---
    def forward(self, claims: List[str], tables: List[Dict[str,Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        roots = [cached_tree(c) for c in claims]
        Hs = []
        for c_root, table in zip(roots, tables):
            pairs, edges = self._collect_pairs(c_root, table)
            self_vecs = self._encode_pairs_with_grad(pairs)      # {nid: [H]}
            h_root = self._compose_from_leaves(0, edges, self_vecs)  # [H]
            Hs.append(h_root.to(device))
        H = torch.stack(Hs, 0)                                   # [B,H]
        return self.head(H)                                      # [B,2]

# --------------- Train / Eval ---------------
@torch.no_grad()
def evaluate(model: Stage2JointLarge, dl, device):
    model.eval()
    total = correct = 0
    for claims, tables, y in tqdm(dl, desc="Validate", leave=False, dynamic_ncols=True):
        y = y.to(device)
        logits = model(claims, tables)
        pred = logits.argmax(-1)
        correct += (pred==y).sum().item()
        total   += y.numel()
    return correct / max(1,total)

def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train", required=False, default="D:\\my_datasets\\tabfact\\train.jsonl")
    ap.add_argument("--dev",   required=False, default="D:\\my_datasets\\tabfact\\validation.jsonl")
    ap.add_argument("--test",  required=False, default="D:\\my_datasets\\tabfact\\test.jsonl")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    # 模型
    ap.add_argument("--model_name", default="microsoft/deberta-v3-large")
    ap.add_argument("--K_self", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--child_method", choices=["attn","mean"], default="attn")
    ap.add_argument("--node_mb_size", type=int, default=64)     # 每样本内节点 micro-batch
    ap.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=False)
    # 优化
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head",     type=float, default=3e-4)  # 门控/投影/Norm/头更大学习率
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    # 运行
    ap.add_argument("--num_workers", type=int, default=0)       # Windows 建议 0
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="stage2_joint_large.pt")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NodeEncoder（注意：不冻结、不eval）
    node = NodeEncoder(model_name=args.model_name, device=device)
    node.enc.train()  # 覆盖 NodeEncoder.__init__ 里的 .eval()
    # 建议 gate 初始更信自分支（对齐基座），可略微稳定训练
    with torch.no_grad():
        if getattr(node, "gate", None) is not None and getattr(node.gate, "bias", None) is not None:
            node.gate.bias.fill_(-1.0)

    model = Stage2JointLarge(
        node, num_labels=2, max_len=args.max_len, K_self=args.K_self,
        child_method=args.child_method, node_mb_size=args.node_mb_size,
        grad_ckpt=args.grad_ckpt
    ).to(device)

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

    # 参数组：encoder vs gate/head
    enc_params = []
    gate_params = []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("node.enc."):
            enc_params.append(p)
        else:
            gate_params.append(p)
    optim = torch.optim.AdamW(
        [{"params": enc_params,  "lr": args.lr_backbone},
         {"params": gate_params, "lr": args.lr_head}],
        weight_decay=args.weight_decay
    )

    # 线性 warmup + 线性余量
    steps_per_epoch = math.ceil(len(train_loader) / max(1,args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    def lr_lambda(step):
        if step < total_steps*args.warmup_ratio:
            return step / max(1.0, total_steps*args.warmup_ratio)
        else:
            remain = step - total_steps*args.warmup_ratio
            return max(0.0, 1.0 - remain / max(1.0, total_steps*(1-args.warmup_ratio)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type=="cuda"))

    print(f"[info] train={len(train_set)}, dev={len(dev_set)}, "
          f"batch={args.batch_size}, accum={args.grad_accum}, "
          f"K_self={args.K_self}, max_len={args.max_len}, node_mb={args.node_mb_size}")

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

            running += loss.item()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0:
                # AMP: 先 unscale 再裁剪
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

            # 显示两个 param group 的 lr
            lr_enc = optim.param_groups[0]["lr"]
            lr_gate= optim.param_groups[1]["lr"]
            pbar.set_postfix(loss=f"{running/max(1,step):.4f}", lr_e=f"{lr_enc:.2e}", lr_g=f"{lr_gate:.2e}",
                             updates=global_updates)

        # 验证
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
