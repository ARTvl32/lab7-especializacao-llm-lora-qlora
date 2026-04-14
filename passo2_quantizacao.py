"""
Laboratório 7 — Passo 2: Configuração da Quantização (QLoRA)
=============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
O treinamento tradicional (Full Fine-Tuning) exige a atualização de todos os
parâmetros do modelo, o que estouraria a memória da GPU para um modelo de
7 bilhões de parâmetros.

A solução é a quantização de 4 bits com a biblioteca bitsandbytes:
o modelo base é carregado com precisão reduzida (NormalFloat 4-bit),
enquanto os cálculos de forward/backward são feitos em float16,
economizando aproximadamente 75% da memória em relação ao float32.

Configuração obrigatória conforme enunciado
-------------------------------------------
    load_in_4bit            = True
    bnb_4bit_quant_type     = "nf4"   (NormalFloat 4-bit)
    bnb_4bit_compute_dtype  = float16
    bnb_4bit_use_double_quant = True  (quantização dupla: reduz ainda mais)

Dependências
------------
    pip install transformers bitsandbytes accelerate torch
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Passo 2 — BitsAndBytesConfig para quantização 4-bit (nf4/float16)
# ---------------------------------------------------------------------------

def criar_bnb_config():
    """
    Cria e retorna a configuração de quantização BitsAndBytes para QLoRA.

    A quantização NF4 (NormalFloat 4-bit) distribui os níveis de quantização
    seguindo uma distribuição normal, o que é mais adequado para os pesos de
    redes neurais que tipicamente seguem distribuições gaussianas.

    Parâmetros do BitsAndBytesConfig
    ---------------------------------
    load_in_4bit              : Carrega o modelo em 4 bits
    bnb_4bit_quant_type       : Tipo de quantização — "nf4" (NormalFloat 4-bit)
    bnb_4bit_compute_dtype    : Dtype dos cálculos — float16 (meia precisão)
    bnb_4bit_use_double_quant : Quantiza também as constantes de quantização,
                                economizando ~0.4 bits adicionais por parâmetro

    Retorna
    -------
    bnb_config : BitsAndBytesConfig
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )
    return bnb_config


def carregar_modelo_base(model_name, bnb_config):
    """
    Carrega o modelo base Llama 2 com a quantização 4-bit configurada.

    Parâmetros
    ----------
    model_name : str               — nome ou caminho do modelo no Hugging Face
    bnb_config : BitsAndBytesConfig

    Retorna
    -------
    model : AutoModelForCausalLM  — modelo quantizado em 4 bits
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",      # distribui automaticamente entre GPU/CPU
        trust_remote_code   = True,
    )

    # Desabilita cache durante treinamento (incompatível com gradient checkpointing)
    model.config.use_cache           = False
    model.config.pretraining_tp      = 1

    return model


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("PASSO 2 — Configuração da Quantização QLoRA (4-bit)")
    print("=" * 65)

    bnb_config = criar_bnb_config()

    print("\nBitsAndBytesConfig instanciado:")
    print(f"  load_in_4bit              = {bnb_config.load_in_4bit}")
    print(f"  bnb_4bit_quant_type       = {bnb_config.bnb_4bit_quant_type}")
    print(f"  bnb_4bit_compute_dtype    = {bnb_config.bnb_4bit_compute_dtype}")
    print(f"  bnb_4bit_use_double_quant = {bnb_config.bnb_4bit_use_double_quant}")

    print("\nNota: Para carregar o modelo completo, execute:")
    print("  model = carregar_modelo_base('meta-llama/Llama-2-7b-hf', bnb_config)")
    print("\n  Requer: acesso ao Hugging Face Hub e GPU com >= 6 GB de VRAM.")

    # Verificar disponibilidade de GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("\n⚠ GPU não detectada. Execute no Google Colab para treinamento completo.")

    print("\n✓ BitsAndBytesConfig configurado corretamente.")
    print("=" * 65)

    return bnb_config


if __name__ == "__main__":
    demo()
