import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "modelo_r32_lr0.0002_ep2"   


tok = AutoTokenizer.from_pretrained(MODELO)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODELO, quantization_config=bnb, device_map={"": 0})
if ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

norm = lambda s: re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()
testes = [json.loads(l) for l in open("teste.jsonl", encoding="utf-8")]

certos, saida = 0, []
for x in testes:
    p = f"### Tabela:\n{x['tabela']}\n### Pergunta:\n{x['pergunta']}\n### SQL:\n"
    ids = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ok = norm(resp) == norm(x["sql"])
    certos += ok
    saida.append({"pergunta": x["pergunta"], "esperado": x["sql"], "gerado": resp, "correto": ok})

print(f"Acertos: {certos}/{len(testes)} = {100*certos/len(testes):.1f}%")
json.dump(saida, open(f"resultados_{ADAPTER}.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)