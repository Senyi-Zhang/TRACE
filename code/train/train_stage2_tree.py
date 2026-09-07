import os, json, math, random, argparse, re
from typing import List, Tuple, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from wordfreq import zipf_frequency

# ---- 你的实现 ----
from NodeEncoder import NodeEncoder
from semantic_tree import claim_to_span_tree
from functools import lru_cache
@lru_cache(maxsize=100000)
def cached_tree(text):
    return claim_to_span_tree(text)

# ---------------- Utils ----------------
def seed_all(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

LABEL_TRUE  = {"1","true","entailed","entails","support","supports","supported","yes","pos","positive","SUPPORTED"}
LABEL_FALSE = {"0","false","refuted","contradiction","contradicts","no","neg","negative","REFUTED"}
def norm_label(x):
    if isinstance(x,(int,float)): return int(round(float(x)))
    s = str(x).strip().lower()
    if s.isdigit(): return int(s)
    if s in {w.lower() for w in LABEL_TRUE}:  return 1
    if s in {w.lower() for w in LABEL_FALSE}: return 0
    raise ValueError(f"Unrecognized label: {x}")

def is_number(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

# ---------------- Data ----------------
class TabFactTreeJsonl(Dataset):
    """
    JSONL 每行支持两种格式之一：
    A) {claim/statement, table:{header, rows, caption}, label}
    B) {claim/statement, header:[...], rows:[[...], ...], caption:str, label}
    """
    def __init__(self, path: str, max_samples: int = None):
        self.recs = []
        ok, skip = 0, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)

                claim = r.get("statement") or r.get("claim")
                label = r.get("label")

                # 1) 优先读取 table 键
                table = r.get("table")

                # 2) 若没有 table，则从顶层 header/rows/caption 组装
                if table is None:
                    header  = r.get("header")
                    rows    = r.get("rows")
                    caption = r.get("caption")

                    # 只要三者之一存在，就尝试构表
                    if header is not None or rows is not None or caption is not None:
                        header  = [] if header is None else list(header)
                        caption = "" if caption is None else str(caption)
                        # 统一成 list[list[str]]
                        if isinstance(rows, list):
                            rows = [[str(x) for x in row] for row in rows]
                        else:
                            rows = []
                        table = {"header": header, "rows": rows, "caption": caption}

                # 3) 校验并收集
                if claim is None or label is None or table is None:
                    skip += 1
                    continue

                try:
                    y = norm_label(label)
                except Exception:
                    skip += 1
                    continue

                self.recs.append((str(claim), table, y))
                ok += 1

                if max_samples and len(self.recs) >= max_samples:
                    break

        assert len(self.recs) > 0, f"No usable samples in {path}. ok={ok}, skip={skip}"

    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]

def collate(batch):
    claims, tables, labels = zip(*batch)
    return list(claims), list(tables), torch.tensor(labels, dtype=torch.long)

# ---------------- Row linearization (与你 Stage-1 一致) ----------------
def linearize_row(table: Dict[str,Any]) -> List[str]:
    rows_out = []
    cap = str(table.get('caption', '') or '')
    header = list(map(str, table.get('header', [])))
    for d in table.get('rows', []):
        title = f"[TITLE]  {cap} |  [ROW] "
        # 小心行长不匹配
        for i in range(min(len(header), len(d))):
            title += f"{header[i]}: {str(d[i])}   "
        rows_out.append(title)
    return rows_out

def rerank(query: str, table: Dict[str,Any], k: int = 10) -> List[str]:
    # 与你给的逻辑一致：数字加大权重 + zipf 稀有度
    q = re.sub(r"[^\w\s]", " ", str(query))
    toks = q.split()
    rarity = {}
    for w in toks:
        rarity[w] = 8 if is_number(w) else 8 - zipf_frequency(w, "en")
    candidates = linearize_row(table)
    scored = []
    for row in candidates:
        s = 0.0
        for w in toks:
            if w in row:
                s += rarity[w]
        scored.append((s, row))
    scored.sort(key=lambda x: x[1], reverse=True)  # 先按字典序稳定
    scored.sort(key=lambda x: x[0], reverse=True)  # 再按分数
    return [r for _, r in scored[:k]]

# ---------------- Stage-2 Tree Model ----------------
class Stage2Tree(nn.Module):
    """
    冻结 enc；训练 node.proj_self/proj_child/gate/norm + 线性头。
    自分支: encode_pair(node.text, "  ||  ".join(rerank(node.text, table, K_self)))
    子分支: 递归 child -> summarize(attn|mean) -> gate 融合
    """
    def __init__(self, node: NodeEncoder, num_labels=2, max_len=256,
                 child_method="attn", K_self=12, joiner="  ||  ",
                 rerank_fn: Callable[[str,Dict[str,Any],int],List[str]] = None):
        super().__init__()
        self.node = node
        self.max_len = max_len
        self.child_method = child_method
        self.K_self = K_self
        self.joiner = joiner
        self.rerank_fn = rerank if rerank_fn is None else rerank_fn
        h = self.node.enc.config.hidden_size
        self.head = nn.Linear(h, num_labels)
        for p in self.node.enc.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_self(self, claim: str, table: Dict[str,Any]) -> torch.Tensor:
        rows = self.rerank_fn(claim, table, k=self.K_self)
        row_text = self.joiner.join(rows)
        return self.node.encode_pair(claim, row_text, max_len=self.max_len)  # [H]

    def summarize_children_trainable(self, child_list: List[torch.Tensor], self_emb: torch.Tensor):
        if not child_list: return None
        dev = self_emb.device
        childs = torch.stack([c.to(dev) for c in child_list], dim=0)  # [K,H]
        if self.child_method == "attn":
            q = F.normalize(self_emb, p=2, dim=-1)    # [H]
            k = F.normalize(childs, p=2, dim=-1)      # [K,H]
            w = torch.softmax(k @ q, dim=0).unsqueeze(-1)  # [K,1]
            ctx = (childs * w).sum(0)
        else:
            ctx = childs.mean(0)
        return F.normalize(ctx, p=2, dim=-1)

    def encode_node_recursive(self, node_dict: Dict[str,Any], table: Dict[str,Any]) -> torch.Tensor:
        child_list = [self.encode_node_recursive(ch, table) for ch in node_dict.get("children", [])]
        self_emb = self.encode_self(node_dict["text"], table)                  # no_grad through enc
        child_ctx = self.summarize_children_trainable(child_list, self_emb)    # trainable
        if child_ctx is None:
            return self_emb
        h_self  = self.node.proj_self(self_emb)
        h_child = self.node.proj_child(child_ctx)
        g = torch.sigmoid(self.node.gate(torch.cat([h_self, h_child], dim=-1)))  # [1]
        fused = g * h_child + (1 - g) * h_self
        out = self.node.norm(self_emb + self.node.dropout(fused))
        return out

    # === 1) 收集整棵树需要编码的 (node_id, text, context) 列表 ===
    def _collect_pairs(self, node_dict, table, joiner, K):
        pairs = []
        edges = {}
        stack = [(0, node_dict)]
        next_id = 1
        id_map = {id(node_dict): 0}

        while stack:
            nid, nd = stack.pop()
            rows = self.rerank_fn(nd["text"], table, k=K)  # ← 直接用 table
            ctx = joiner.join(rows)
            pairs.append((nid, nd["text"], ctx))

            ch_ids = []
            for ch in nd.get("children", []):
                cid = id_map.get(id(ch))
                if cid is None:
                    cid = next_id;
                    next_id += 1
                    id_map[id(ch)] = cid
                ch_ids.append(cid)
                stack.append((cid, ch))
            edges[nid] = ch_ids
        return pairs, edges

    # === 2) 一次性编码所有 pairs（encoder 冻结+no_grad） ===
    @torch.no_grad()
    def _encode_pairs_once(self, pairs):
        device = next(self.parameters()).device
        texts_a = [p[1] for p in pairs]
        texts_b = [p[2] for p in pairs]
        tok = self.node.tok(
            texts_a, texts_b, truncation=True, max_length=self.max_len,
            padding=True, return_tensors="pt"
        ).to(device)
        h = self.node.enc(**tok).last_hidden_state  # [N,L,H]
        m = tok["attention_mask"].unsqueeze(-1).float()  # [N,L,1]
        vec = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)  # [N,H]
        vec = F.normalize(vec, p=2, dim=-1)
        # id -> [H]
        ids = [p[0] for p in pairs]
        return {i: v for i, v in zip(ids, vec)}

    # === 3) 用预编码的 self 向量做递归融合（只有这部分有梯度） ===
    def _compose_from_leaves(self, nid, edges, self_vecs):
        self_emb = self_vecs[nid]  # [H], no grad
        child_ids = edges.get(nid, [])
        child_list = [self._compose_from_leaves(cid, edges, self_vecs) for cid in child_ids]
        if not child_list:
            return self_emb
        childs = torch.stack(child_list, 0)
        q = F.normalize(self_emb, p=2, dim=-1)
        k = F.normalize(childs, p=2, dim=-1)
        w = torch.softmax(k @ q, dim=0).unsqueeze(-1)
        ctx = (childs * w).sum(0)
        h_self = self.node.proj_self(self_emb)
        h_child = self.node.proj_child(ctx)
        g = torch.sigmoid(self.node.gate(torch.cat([h_self, h_child], dim=-1)))
        fused = g * h_child + (1 - g) * h_self
        return self.node.norm(self_emb + self.node.dropout(fused))

    # === 4) forward：每样本只跑一次 encoder ===
    def forward(self, claims: List[str], tables: List[Dict[str, Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        Hs = []
        for claim, table in zip(claims, tables):
            root = cached_tree(claim)
            pairs, edges = self._collect_pairs(root, table, self.joiner, self.K_self)
            self_vecs = self._encode_pairs_once(pairs)
            h_root = self._compose_from_leaves(0, edges, self_vecs)
            Hs.append(h_root.to(device))
        H = torch.stack(Hs, 0)
        return self.head(H)


# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(model: Stage2Tree, dl, device):
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
    ap.add_argument("--train", default="D:\my_datasets\\tabfact\\train.jsonl")
    ap.add_argument("--dev",   default="D:\my_datasets\\tabfact\\validation.jsonl")
    ap.add_argument("--test",  default="D:\my_datasets\\tabfact\\test.jsonl")
    ap.add_argument("--stage1_ckpt", default="stage1_deberta_baseline.pt")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--K_self", type=int, default=8)
    ap.add_argument("--child_method", choices=["attn","mean"], default="attn")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)     # 树递归较慢
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="stage2_tree.pt")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--max_train_samples", type=int, default=None)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- NodeEncoder & load Stage-1 enc ----
    node = NodeEncoder(model_name=args.model_name, device=device)
    sd = torch.load(args.stage1_ckpt, map_location="cpu")["state_dict"]
    enc_sd = {k[len("enc."):]: v for k,v in sd.items() if k.startswith("enc.")}
    res = node.enc.load_state_dict(enc_sd, strict=False)
    print("[load enc] missing:", res.missing_keys, "| unexpected:", res.unexpected_keys)
    node.enc.eval()
    for p in node.enc.parameters(): p.requires_grad = False

    model = Stage2Tree(node, num_labels=2, max_len=args.max_len,
                       child_method=args.child_method, K_self=args.K_self).to(device)

    # ---- Data ----
    train_full = TabFactTreeJsonl(args.train, max_samples=args.max_train_samples)
    if args.dev:
        dev_set = TabFactTreeJsonl(args.dev)
        train_set = train_full
    else:
        recs = train_full.recs[:]
        random.shuffle(recs)
        n_val = max(1, int(len(recs) * args.val_ratio))
        dev_recs = recs[:n_val]; train_recs = recs[n_val:]
        class ListDS(Dataset):
            def __init__(self, recs): self.recs = recs
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

    # ---- Optim & sched ----
    params = []
    params += list(model.node.proj_self.parameters())
    params += list(model.node.proj_child.parameters())
    params += list(model.node.gate.parameters())
    params += list(model.node.norm.parameters())
    params += list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1,args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_ratio
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type=="cuda"))

    # ---- Train ----
    print(f"[info] train={len(train_set)}, dev={len(dev_set)}, batch={args.batch_size}, accum={args.grad_accum}, K_self={args.K_self}")
    best_dev, best_state = 0.0, None
    for ep in range(1, args.epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", dynamic_ncols=True)
        running = 0.0; updates = 0
        for step, (claims, tables, y) in enumerate(bar, start=1):
            y = y.to(device)
            with torch.amp.autocast("cuda", enabled=(args.fp16 and device.type=="cuda")):
                logits = model(claims, tables)
                loss = F.cross_entropy(logits, y)
            running += loss.item()

            if args.fp16 and device.type=="cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if args.fp16 and device.type=="cuda":
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step(); updates += 1
                opt.zero_grad(set_to_none=True)

            bar.set_postfix(loss=f"{running/step:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}", updates=updates)

        dev_acc = evaluate(model, dev_loader, device)
        print(f"Epoch {ep} | dev acc {dev_acc:.4f}")
        if dev_acc > best_dev + 1e-4:
            best_dev = dev_acc
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args), "model_name": args.model_name}, args.ckpt)
            print("  (saved best)")

    # ---- Test ----
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
