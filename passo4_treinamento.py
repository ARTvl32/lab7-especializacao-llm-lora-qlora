"""
Laboratório 7 — Passo 4: Pipeline de Treinamento e Otimização
=============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Orquestra o pipeline completo de fine-tuning QLoRA usando o SFTTrainer da
biblioteca trl (Transformer Reinforcement Learning), integrando os componentes
dos Passos 1, 2 e 3.

Configurações obrigatórias do enunciado
---------------------------------------
    otimizador         : paged_adamw_32bit
    lr_scheduler_type  : cosine
    warmup_ratio       : 0.03

Pipeline completo
-----------------
    1. Carregar dataset (.jsonl do Passo 1)
    2. Carregar tokenizador do modelo base
    3. Carregar modelo base com quantização 4-bit (Passo 2)
    4. Aplicar adaptadores LoRA ao modelo (Passo 3)
    5. Configurar TrainingArguments com paged_adamw_32bit e cosine scheduler
    6. Instanciar SFTTrainer e executar trainer.train()
    7. Salvar o modelo adaptador com trainer.model.save_pretrained()

Dependências
------------
    pip install transformers peft trl bitsandbytes accelerate datasets torch
"""

import os
from datasets             import load_dataset
from transformers         import AutoTokenizer, TrainingArguments
from trl                  import SFTTrainer
from passo2_quantizacao   import criar_bnb_config, carregar_modelo_base
from passo3_lora_config   import criar_lora_config, aplicar_lora, contar_parametros


# ---------------------------------------------------------------------------
# Configurações do treinamento
# ---------------------------------------------------------------------------

MODEL_NAME       = "meta-llama/Llama-2-7b-hf"   # requer acesso no HF Hub
TRAIN_FILE       = "data/dataset_treino.jsonl"
OUTPUT_DIR       = "lora_adapter"
ADAPTER_SAVE_DIR = "lora_adapter_final"

# Hiperparâmetros de treinamento
NUM_EPOCHS          = 3
BATCH_SIZE          = 4
GRAD_ACCUM_STEPS    = 4          # batch efetivo = 4 × 4 = 16
LEARNING_RATE       = 2e-4
MAX_SEQ_LENGTH      = 512
LOGGING_STEPS       = 10
SAVE_STEPS          = 50


# ---------------------------------------------------------------------------
# Passo 4.1 — Carregar dataset e tokenizador
# ---------------------------------------------------------------------------

def carregar_dataset(train_file=TRAIN_FILE):
    """
    Carrega o dataset .jsonl gerado no Passo 1.

    O SFTTrainer espera um campo de texto único por amostra.
    Formata cada par como:
        ### Instrução:\n{prompt}\n\n### Resposta:\n{response}

    Parâmetros
    ----------
    train_file : str  — caminho para o arquivo .jsonl de treino

    Retorna
    -------
    dataset : Dataset (Hugging Face)
    """
    dataset = load_dataset("json", data_files={"train": train_file}, split="train")

    def formatar_prompt(sample):
        """Formata o par instrução/resposta no template Alpaca."""
        texto = (
            f"### Instrução:\n{sample['prompt']}\n\n"
            f"### Resposta:\n{sample['response']}"
        )
        return {"text": texto}

    dataset = dataset.map(formatar_prompt)
    return dataset


def carregar_tokenizador(model_name=MODEL_NAME):
    """
    Carrega o tokenizador do modelo base.

    Parâmetros
    ----------
    model_name : str

    Retorna
    -------
    tokenizer : AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # evita warnings durante treinamento
    return tokenizer


# ---------------------------------------------------------------------------
# Passo 4.2 — TrainingArguments com otimizador e scheduler obrigatórios
# ---------------------------------------------------------------------------

def criar_training_args(output_dir=OUTPUT_DIR):
    """
    Configura os argumentos de treinamento com as opções obrigatórias:
        - Otimizador: paged_adamw_32bit
        - LR Scheduler: cosine
        - Warmup Ratio: 0.03

    O paged_adamw_32bit usa paginação para transferir picos de memória do
    AdamW (estados do otimizador: momentum e variância) da GPU para a RAM,
    permitindo treinar modelos maiores em GPUs com menos VRAM.

    O scheduler cosine decai o learning rate seguindo uma curva de cosseno,
    evitando quedas bruscas e suavizando a convergência.

    O warmup_ratio=0.03 faz o LR subir gradualmente nos primeiros 3% dos
    passos de treinamento, evitando instabilidades no início.

    Retorna
    -------
    training_args : TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE,

        # --- Configurações obrigatórias do enunciado ---
        optim               = "paged_adamw_32bit",
        lr_scheduler_type   = "cosine",
        warmup_ratio        = 0.03,
        # -----------------------------------------------

        fp16                        = True,
        gradient_checkpointing      = True,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,
        report_to                   = "none",      # desativa W&B/TensorBoard
    )
    return training_args


# ---------------------------------------------------------------------------
# Passo 4.3 — Pipeline completo: SFTTrainer + train() + save_pretrained()
# ---------------------------------------------------------------------------

def executar_treinamento():
    """
    Executa o pipeline completo de fine-tuning QLoRA:

        Dataset  →  Tokenizador  →  Modelo 4-bit  →  LoRA
        →  SFTTrainer  →  train()  →  save_pretrained()
    """
    print("=" * 65)
    print("PASSO 4 — Pipeline de Treinamento (SFTTrainer + QLoRA)")
    print("=" * 65)

    # 1. Dataset
    print("\n[1/6] Carregando dataset...")
    dataset   = carregar_dataset()
    print(f"  ✓ {len(dataset)} amostras carregadas de '{TRAIN_FILE}'")

    # 2. Tokenizador
    print("\n[2/6] Carregando tokenizador...")
    tokenizer = carregar_tokenizador()
    print(f"  ✓ Tokenizador carregado: vocab_size={tokenizer.vocab_size:,}")

    # 3. Modelo base quantizado
    print("\n[3/6] Carregando modelo base com quantização 4-bit...")
    bnb_config = criar_bnb_config()
    model      = carregar_modelo_base(MODEL_NAME, bnb_config)
    print(f"  ✓ Modelo base carregado em 4-bit (nf4/float16)")

    # 4. Aplicar LoRA
    print("\n[4/6] Aplicando adaptadores LoRA...")
    lora_config = criar_lora_config()
    model       = aplicar_lora(model, lora_config)
    treinavel, total, pct = contar_parametros(model)
    print(f"  ✓ Parâmetros treináveis : {treinavel:,} / {total:,} ({pct:.2f}%)")

    # 5. TrainingArguments
    print("\n[5/6] Configurando TrainingArguments...")
    training_args = criar_training_args()
    print(f"  ✓ Otimizador          : paged_adamw_32bit")
    print(f"  ✓ LR Scheduler        : cosine")
    print(f"  ✓ Warmup Ratio        : 0.03")
    print(f"  ✓ Learning Rate       : {LEARNING_RATE}")
    print(f"  ✓ Épocas              : {NUM_EPOCHS}")
    print(f"  ✓ Batch efetivo       : {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # 6. SFTTrainer
    print("\n[6/6] Iniciando treinamento com SFTTrainer...")
    trainer = SFTTrainer(
        model           = model,
        train_dataset   = dataset,
        tokenizer       = tokenizer,
        args            = training_args,
        dataset_text_field = "text",
        max_seq_length  = MAX_SEQ_LENGTH,
        packing         = False,
    )

    trainer.train()

    # 7. Salvar adaptador LoRA (conforme enunciado: trainer.model.save_pretrained)
    print(f"\n✓ Salvando modelo adaptador em '{ADAPTER_SAVE_DIR}'...")
    trainer.model.save_pretrained(ADAPTER_SAVE_DIR)
    tokenizer.save_pretrained(ADAPTER_SAVE_DIR)
    print(f"✓ Adaptador LoRA salvo com sucesso.")

    print("=" * 65)
    return trainer


# ---------------------------------------------------------------------------
# Demonstração (exibe configurações sem executar o treinamento)
# ---------------------------------------------------------------------------

def demo_config():
    """
    Exibe todas as configurações do pipeline sem carregar o modelo completo.
    Útil para verificar as configurações antes de executar no Colab/GPU.
    """
    print("=" * 65)
    print("PASSO 4 — Configurações do Pipeline de Treinamento")
    print("=" * 65)

    bnb_config    = criar_bnb_config()
    lora_config   = criar_lora_config()
    training_args = criar_training_args()

    print("\n--- Modelo Base ---")
    print(f"  {MODEL_NAME}")

    print("\n--- Quantização (Passo 2) ---")
    print(f"  load_in_4bit           = {bnb_config.load_in_4bit}")
    print(f"  bnb_4bit_quant_type    = {bnb_config.bnb_4bit_quant_type}")
    print(f"  bnb_4bit_compute_dtype = {bnb_config.bnb_4bit_compute_dtype}")

    print("\n--- LoRA (Passo 3) ---")
    print(f"  r (rank)     = {lora_config.r}")
    print(f"  lora_alpha   = {lora_config.lora_alpha}")
    print(f"  lora_dropout = {lora_config.lora_dropout}")
    print(f"  task_type    = {lora_config.task_type}")

    print("\n--- TrainingArguments (Passo 4) ---")
    print(f"  optim              = {training_args.optim}")
    print(f"  lr_scheduler_type  = {training_args.lr_scheduler_type}")
    print(f"  warmup_ratio       = {training_args.warmup_ratio}")
    print(f"  learning_rate      = {training_args.learning_rate}")
    print(f"  num_train_epochs   = {training_args.num_train_epochs}")
    print(f"  fp16               = {training_args.fp16}")

    print("\n✓ Para executar o treinamento completo, chame executar_treinamento()")
    print("  (requer GPU com >= 6 GB VRAM e acesso ao meta-llama/Llama-2-7b-hf)")
    print("=" * 65)


if __name__ == "__main__":
    demo_config()
