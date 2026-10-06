import torch
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "modelo_r32_lr0.0002_ep2"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

# Configuração 4-bit para caber na RTX 2060
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Carrega o Qwen original
modelo = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# Aplica nosso fine-tuning LoRA
modelo = PeftModel.from_pretrained(modelo, ADAPTER)
modelo.eval()


def gerar_sql(tabela, pergunta):

    prompt = f"""### Tabela:
{tabela}
### Pergunta:
{pergunta}
### SQL:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    resposta = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Retorna apenas o que veio depois de ### SQL:
    sql = resposta.split("### SQL:")[-1].strip()

    return sql


interface = gr.Interface(
    fn=gerar_sql,
    inputs=[
        gr.Textbox(
            label="Estrutura da tabela",
            placeholder="Ex: funcionarios(id, nome, salario, departamento)"
        ),
        gr.Textbox(
            label="Pergunta",
            placeholder="Ex: Quais são os 5 funcionários com maior salário?"
        )
    ],
    outputs=gr.Code(
        label="SQL gerado",
        language="sql"
    ),
    title="Gerador de SQL com LLM",
    description="Qwen2.5-1.5B-Instruct treinado com QLoRA para geração de consultas SQL."
)

interface.launch()