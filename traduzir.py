import json, torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

N = 1000
ds = load_dataset("b-mc2/sql-create-context", split=f"train[:{N}]")

nome = "Helsinki-NLP/opus-mt-en-ROMANCE"
tok = MarianTokenizer.from_pretrained(nome)
model = MarianMTModel.from_pretrained(nome).half().cuda()

perguntas = [">>pt_br<< " + q for q in ds["question"]]
saida = []
for i in range(0, N, 32):
    lote = tok(perguntas[i:i+32], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        ids = model.generate(**lote, max_length=128)
    saida += tok.batch_decode(ids, skip_special_tokens=True)
    print(i)

with open("dataset_sql_pt.jsonl", "w", encoding="utf-8") as f:
    for q, c, a in zip(saida, ds["context"], ds["answer"]):
        f.write(json.dumps({"pergunta": q, "tabela": c, "sql": a}, ensure_ascii=False) + "\n")