# your_infer_script.py

import os, importlib.util
from wordfreq import zipf_frequency
import re
# 1) 指定 plan_B.py 的绝对路径
PLANB_FILE = r"D:\desktop\Table-FC\my_retrieval\plan_B.py"  # ← 改成你的实际路径

# 2) 动态加载模块，不依赖工作目录
spec = importlib.util.spec_from_file_location("plan_B", PLANB_FILE)
plan_B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_B)

# 3) 用 plan_B.<函数/常量> 访问
tok, model, device = plan_B.load_reranker(plan_B.OUTPUT_DIR)

sbert_index, table_ids, table_db = plan_B.load_sbert_index(
    path_table_json=plan_B.PATH_TABLE_JSON,
    sbert_model_name=plan_B.SBERT_MODEL_NAME,
    cache_dir=plan_B.SBERT_CACHE_DIR,
)

def retrieve(query, top_k=10, recall=500):
    return plan_B.rerank_query_uids(
        query=query,
        sbert_index=sbert_index,
        table_db=table_db,
        reranker_tok=tok,
        reranker_model=model,
        topk_return=top_k,
        topk_recall=recall,
    )

def linearize_row(table):
    rows = []
    for d in table['data']:
        title = '[TITLE] ' + table['title'] + '  |  [ROW] '
        i = 0
        while i < len(table['header']):
            title += table['header'][i] + ': ' + d[i] + '   '
            i += 1
        rows.append(title)
    return rows

def rerank(query, tables, k=10):
    query = re.sub(r'[^\w\s]', ' ', query)
    words = query.split()
    rarity_dict = {}
    lis = []
    for word in words:
        rarity_dict[word] = 8 - zipf_frequency(word, 'en')
    all_rows = []
    for table in tables:
        all_rows.extend(linearize_row(table))
    for row in all_rows:
        score = 0
        for word in words:
            if word in row:
                score += rarity_dict[word]
        lis.append([row, score])
    sorted_data_desc = sorted(lis, key=lambda x: x[1], reverse=True)
    return [entry[0] for entry in sorted_data_desc[:k]]
