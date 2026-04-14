# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

**Disciplina:** Tópicos em Inteligência Artificial 2026.1
**Instituição:** iCEV — Instituto de Ensino Superior
**Professor:** Dimmy Magalhães

> **Nota sobre uso de IA Generativa:** Partes geradas/complementadas com IA,
> revisadas por Arthur. Especificamente: a geração dos prompts do dataset
> sintético (Passo 1) utilizou IA para brainstorming dos pares
> pergunta/resposta. A estrutura dos scripts, as configurações de LoRA/QLoRA
> e o pipeline de treinamento foram implementados e revisados manualmente com
> base nas aulas e na documentação oficial das bibliotecas.

---

## Objetivo

Construir um pipeline completo de *fine-tuning* de um modelo de linguagem
fundacional (Llama 2 7B) utilizando técnicas de eficiência de parâmetros
(**PEFT/LoRA**) e quantização (**QLoRA**) para viabilizar o treinamento em
hardwares limitados.

---

## Estrutura do Repositório

```
lab7-lora/
│
├── passo1_gerar_dataset.py     # Geração do dataset sintético via API OpenAI
├── passo2_quantizacao.py       # Configuração BitsAndBytes 4-bit (nf4/float16)
├── passo3_lora_config.py       # Arquitetura LoRA (r=64, alpha=16, dropout=0.1)
├── passo4_treinamento.py       # SFTTrainer + paged_adamw_32bit + cosine LR
├── data/
│   ├── dataset_treino.jsonl    # 90% dos pares gerados (≥ 45 pares)
│   └── dataset_teste.jsonl     # 10% dos pares gerados (≥ 5 pares)
└── README.md
```

---

## Como Executar

> **Recomendado:** Google Colab com GPU T4 (gratuito) ou A100 (Colab Pro).
> O Llama 2 7B em 4-bit exige aproximadamente 6–8 GB de VRAM.

```bash
# Instalar dependências
pip install openai datasets transformers peft trl bitsandbytes accelerate

# Passo 1 — Gerar dataset sintético (requer OPENAI_API_KEY no ambiente)
python passo1_gerar_dataset.py

# Passo 2 — Testar configuração de quantização
python passo2_quantizacao.py

# Passo 3 — Testar configuração LoRA
python passo3_lora_config.py

# Passo 4 — Executar pipeline de treinamento completo
python passo4_treinamento.py
```

---

## Passo 1 — Dataset Sintético

O script `passo1_gerar_dataset.py` usa a API da OpenAI (GPT-3.5-turbo) para
gerar **50 pares** de instrução/resposta no domínio de **Engenharia de
Software**. Os dados são divididos em:

- `dataset_treino.jsonl` — 90% dos pares (45 amostras)
- `dataset_teste.jsonl`  — 10% dos pares (5 amostras)

Formato de cada linha `.jsonl`:
```json
{"prompt": "O que é refatoração de código?", "response": "Refatoração é o processo de..."}
```

---

## Passo 2 — Quantização QLoRA (4-bit)

Configuração do `BitsAndBytesConfig`:

| Parâmetro             | Valor    | Justificativa                            |
|-----------------------|----------|------------------------------------------|
| `load_in_4bit`        | `True`   | Reduz uso de VRAM ~4×                    |
| `bnb_4bit_quant_type` | `"nf4"`  | NormalFloat 4-bit: melhor para pesos LLM |
| `bnb_4bit_compute_dtype` | `float16` | Cálculos em meia precisão            |
| `bnb_4bit_use_double_quant` | `True` | Quantização dupla: reduz ainda mais |

---

## Passo 3 — Arquitetura LoRA

O LoRA **congela** a matriz original $W$ e injeta duas matrizes menores
$A$ e $B$ de decomposição de baixo posto (*low-rank*):

$$W' = W + \frac{\alpha}{r} \cdot BA$$

| Hiperparâmetro | Valor | Papel                                          |
|----------------|-------|------------------------------------------------|
| `r` (rank)     | 64    | Dimensão das matrizes menores A e B            |
| `lora_alpha`   | 16    | Fator de escala dos novos pesos (α/r = 0.25)   |
| `lora_dropout` | 0.1   | Regularização para evitar overfitting          |
| `task_type`    | `CAUSAL_LM` | Tarefa de geração de linguagem causal  |

---

## Passo 4 — Pipeline de Treinamento

Configuração do `TrainingArguments`:

| Parâmetro             | Valor              | Justificativa                               |
|-----------------------|--------------------|---------------------------------------------|
| `optim`               | `paged_adamw_32bit`| Transfere picos de memória GPU → CPU        |
| `lr_scheduler_type`   | `cosine`           | LR decai suavemente em curva cosseno        |
| `warmup_ratio`        | `0.03`             | Primeiros 3% do treino: LR sobe gradualmente|
| `learning_rate`       | `2e-4`             | LR base para fine-tuning com LoRA           |
| `num_train_epochs`    | `3`                | Número de épocas de treinamento             |
| `per_device_train_batch_size` | `4`      | Batch por dispositivo                       |
| `gradient_accumulation_steps` | `4`      | Acumula gradientes para batch efetivo de 16 |
| `fp16`                | `True`             | Treinamento em meia precisão                |

---

## Fundamentos Matemáticos

**LoRA — Decomposição de baixo posto:**

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

onde $W_0 \in \mathbb{R}^{d \times k}$ é a matriz congelada,
$A \in \mathbb{R}^{r \times k}$ e $B \in \mathbb{R}^{d \times r}$
são as matrizes treináveis com $r \ll \min(d, k)$.

**QLoRA — Quantização 4-bit com NormalFloat:**

O NF4 distribui os níveis de quantização seguindo uma distribuição normal
(ao invés de uniforme), o que é mais adequado para os pesos de redes neurais
que tipicamente seguem distribuições gaussianas.

---

## Referências

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Notas de aula — Prof. Dimmy Magalhães, iCEV 2026.1
