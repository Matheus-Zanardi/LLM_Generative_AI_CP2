import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODELO = "Qwen/Qwen2.5-1.5B-Instruct"

R = 32
LR = 2e-4
EPOCAS = 2

SAIDA = f"modelo_r{R}_lr{LR}_ep{EPOCAS}"

tok = AutoTokenizer.from_pretrained(MODELO)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODELO, quantization_config=bnb, device_map={"": 0})
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=2*R, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))

ds = load_dataset("json", data_files="dataset_sql_pt.jsonl", split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)
ds["test"].to_json("teste.jsonl", force_ascii=False)

def fmt(x):
    t = (f"### Tabela:\n{x['tabela']}\n### Pergunta:\n{x['pergunta']}\n"
         f"### SQL:\n{x['sql']}{tok.eos_token}")
    return tok(t, truncation=True, max_length=256)

treino = ds["train"].map(fmt, remove_columns=ds["train"].column_names)

args = TrainingArguments(output_dir=SAIDA, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=EPOCAS, learning_rate=LR,
        fp16=True, logging_steps=10, save_strategy="no", report_to="none",
        optim="paged_adamw_8bit")

Trainer(model=model, args=args, train_dataset=treino,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

model.save_pretrained(SAIDA)
tok.save_pretrained(SAIDA)