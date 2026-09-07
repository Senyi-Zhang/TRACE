from tabfact_pipeline import rerank
from TabFact.download_tabfact import get_train_data
from tqdm import tqdm
from json_utils import write_jsonl
data = get_train_data('validation')
write_data = []
for d in tqdm(data):
    rows = rerank(d['statement'], d)
    d['rerank_rows'] = rows
    write_data.append(d)
    if len(write_data) >= 200:
        write_jsonl('D:\my_datasets\\tabfact\\validation.jsonl', write_data)
        write_data = []
