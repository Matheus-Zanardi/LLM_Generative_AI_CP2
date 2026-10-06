# Gerador de SQL com LLM Local

## Integrantes do Grupo
- Augusto Fisco Milreu - RM98245
- David Denunci - RM98603
- Fernando Popolili - RM99919
- Matheus Zanardi - RM98832
- Lucas Palamartschuk de Toledo - RM97913

Projeto desenvolvido para o **CP2/CP3 - Generative AI for Engineering**, com o objetivo de realizar fine-tuning de uma LLM local utilizando GPU e disponibilizar o modelo por meio de uma interface gráfica.

## Objetivo

O projeto utiliza uma Large Language Model para transformar perguntas em linguagem natural, em português, em consultas SQL.

### Exemplo

Entrada:

```text
Tabela:
funcionarios(id, nome, salario, departamento)

Pergunta:
Quais são os 5 funcionários com maior salário?
```

Saída:

```sql
SELECT id, nome, salario
FROM funcionarios
ORDER BY salario DESC
LIMIT 5;
```

## Modelo utilizado

Foi utilizado o modelo:

**Qwen2.5-1.5B-Instruct**

O modelo foi escolhido por possuir tamanho relativamente pequeno, permitindo execução e fine-tuning local em uma GPU com recursos limitados.

## Hardware

O treinamento foi realizado utilizando GPU NVIDIA.

- GPU: NVIDIA GeForce RTX 2060
- Treinamento realizado com CUDA
- Precisão: FP16
- Quantização: 4-bit

A utilização da GPU durante o projeto foi verificada utilizando `nvidia-smi` e PyTorch.

## Fine-tuning

Para adaptar o modelo à geração de SQL foi utilizado **QLoRA**.

A quantização em 4 bits reduz o consumo de VRAM, enquanto LoRA permite treinar uma quantidade reduzida de parâmetros do modelo.

Bibliotecas principais:

- PyTorch
- Transformers
- PEFT
- BitsAndBytes
- Datasets
- Gradio

## Dataset

O dataset utilizado contém exemplos formados por:

- estrutura da tabela;
- pergunta em português;
- consulta SQL esperada.

Formato utilizado durante o treinamento:

```text
### Tabela:
funcionarios(id, nome, salario, departamento)

### Pergunta:
Quais são os funcionários do departamento de TI?

### SQL:
SELECT * FROM funcionarios WHERE departamento = 'TI';
```

O dataset completo contém **1.000 exemplos** de pares pergunta–SQL.

Para o treinamento e avaliação, os dados foram divididos utilizando `train_test_split` com `test_size=0.1` e `seed=42`, garantindo um split reproduzível:

- **900 exemplos (90%)** utilizados no treinamento;
- **100 exemplos (10%)** reservados exclusivamente para avaliação.

O conjunto de teste foi salvo no arquivo `teste.jsonl` e utilizado para comparar o desempenho do modelo base com os modelos após o fine-tuning.

## Experimentos

Foram realizados diferentes treinamentos alterando hiperparâmetros do LoRA e do processo de treinamento.

| Modelo | LoRA Rank | Learning Rate | Épocas | Acurácia |
|---|---:|---:|---:|---:|
| Modelo base | - | - | - | 6% |
| Teste A | 16 | 2e-4 | 2 | 48% |
| Teste B | 8 | 1e-4 | 2 | 42% |
| Teste C | 32 | 2e-4 | 2 | 53% |

O melhor resultado foi obtido pelo **Teste C**, com **53% de acurácia**.

O fine-tuning aumentou a acurácia de **6% para 53%**, representando uma melhora de **47 pontos percentuais** no conjunto de avaliação utilizado.

## Avaliação

A avaliação automática foi realizada sobre 100 exemplos separados para teste.

Para cada exemplo, o modelo recebeu a estrutura da tabela e a pergunta em linguagem natural. O SQL gerado foi comparado com o SQL esperado no dataset.

Antes da comparação foram normalizados:

- espaços em branco;
- letras maiúsculas e minúsculas;
- ponto e vírgula ao final da consulta.

A métrica utilizada foi **Exact Match Accuracy**. Portanto, uma resposta somente é considerada correta quando o SQL gerado corresponde ao SQL de referência após essas normalizações.

Essa é uma métrica rigorosa, pois consultas SQL semanticamente equivalentes podem ser classificadas como diferentes caso utilizem outra estrutura textual.

Além da avaliação automática, foram realizados testes manuais através do frontend.

### Teste 1

Pergunta:

```text
Quais são os 5 funcionários com maior salário?
```

Resposta:

```sql
SELECT id, nome, salario
FROM funcionarios
ORDER BY salario DESC
LIMIT 5;
```

### Teste 2

Pergunta:

```text
Quantos clientes existem em cada cidade?
```

Resposta:

```sql
SELECT COUNT(*), cidade
FROM clientes
GROUP BY cidade;
```

### Teste 3

Pergunta:

```text
Qual é o valor médio das vendas da categoria Eletrônicos?
```

Resposta gerada:

```sql
SELECT AVG(valor)
FROM vendas
WHERE categoria = 'Eletrônico';
```

Nesse último exemplo, o modelo identificou corretamente a necessidade de utilizar `AVG` e `WHERE`, porém modificou o valor literal de `Eletrônicos` para `Eletrônico`.

Esse comportamento demonstra uma das limitações encontradas durante os testes.

## Frontend

Foi desenvolvido um frontend utilizando **Gradio**.

A interface permite informar:

1. estrutura da tabela;
2. pergunta em linguagem natural.

O modelo processa essas informações localmente e retorna a consulta SQL gerada.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o frontend:

```bash
python app.py
```

Depois acesse o endereço local informado pelo Gradio, normalmente:

```text
http://127.0.0.1:7860
```

## Estrutura do projeto

```text
.
├── app.py
├── avaliar.py
├── treinar.py
├── traduzir.py
├── dataset_sql_pt.jsonl
├── teste.jsonl
├── requirements.txt
├── modelo_r8_lr0.0001_ep2/
├── modelo_r16_lr0.0002_ep2/
├── modelo_r32_lr0.0002_ep2/
└── resultados_*.json
```

## Tecnologias

Python, PyTorch, Hugging Face Transformers, PEFT, QLoRA, BitsAndBytes, Gradio e CUDA.

## Conclusão

Os experimentos demonstraram que o fine-tuning especializado melhorou significativamente o desempenho do modelo na tarefa de geração de SQL.

O modelo base obteve 6% de acurácia, enquanto a melhor configuração treinada atingiu 53%.

Entre as configurações testadas, o LoRA Rank 32 com learning rate `2e-4` e 2 épocas apresentou o melhor resultado.

O projeto também demonstra a possibilidade de executar e adaptar uma LLM local utilizando uma GPU de consumidor, além de disponibilizar o modelo através de uma interface simples para o usuário.
