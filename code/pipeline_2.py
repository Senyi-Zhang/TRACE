from semantic_tree import claim_to_span_tree
from NodeEncoderPMAFiLM import NodeEncoderPMAFiLM
from retrieval.planB_infer import retrieve, get_table, rerank
import json
import torch
from json_utils import read_jsonl
import threading
node_encoder = NodeEncoderPMAFiLM()


def run(text):
    root = claim_to_span_tree(text)
    return post_order(root)


def post_order(node):
    child_list = []
    for child in node["children"]:
        child_emb = post_order(child)
        child_list.append(child_emb)
    uids = retrieve(node['text'])
    tables = [get_table(uid) for uid in uids]
    rows = rerank(node['text'], tables)
    row_text = '  '.join(rows)
    return node_encoder.forward_once(node['text'], row_text, child_list)


def save_hroot_jsonl(path: str, t: torch.Tensor, label = None, extra = None):
    t = t.detach().float().cpu().view(-1)
    vec = t.tolist()

    rec = {"h_root": vec}
    if label is not None:
        rec["label"] = label
    if extra:
        rec.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_file(file_path, a, label='REFUTED'):
    i = 0
    data = read_jsonl(file_path)
    while i < 200:

        if i < len(data):
            line = data[i]
            claim = line['claim']
            tensor = run(claim)
            save_hroot_jsonl(f"train_mlp_dev{a}.jsonl", tensor, label)
            i += 1


if __name__ == "__main__":
    threads = []
    for i in range(4, 6):
        t = threading.Thread(target=process_file, args=(f"D:\desktop\Table-FC\datasets\dev{i}.jsonl", i, ))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()