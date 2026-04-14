"""
Laboratório 7 — Passo 3: Arquitetura do LoRA
============================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
O LoRA (Low-Rank Adaptation) congela todos os parâmetros do modelo base e
injeta pares de matrizes treináveis de baixo posto (low-rank) nas camadas de
atenção. Apenas essas matrizes menores são atualizadas durante o treinamento,
reduzindo drasticamente o número de parâmetros treináveis.

Matemática do LoRA
------------------
Para uma camada linear com peso W ∈ ℝ^(d×k), o LoRA adiciona:

    W' = W + (alpha / r) * B @ A

onde:
    A ∈ ℝ^(r×k)  — matriz de projeção descendente (inicializada aleatoriamente)
    B ∈ ℝ^(d×r)  — matriz de projeção ascendente  (inicializada com zeros)
    r             — rank (dimensão do espaço latente de baixo posto)
    alpha         — fator de escala (alpha/r controla a magnitude do update)

Hiperparâmetros obrigatórios (conforme enunciado)
-------------------------------------------------
    r (rank)      : 64   — dimensão das matrizes menores A e B
    lora_alpha    : 16   — fator de escala dos novos pesos
    lora_dropout  : 0.1  — dropout para evitar overfitting
    task_type     : CAUSAL_LM

Dependências
------------
    pip install peft transformers torch
"""

from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Passo 3 — LoraConfig com hiperparâmetros obrigatórios
# ---------------------------------------------------------------------------

def criar_lora_config():
    """
    Instancia o LoraConfig com os hiperparâmetros obrigatórios do enunciado.

    Parâmetros do LoraConfig
    ------------------------
    r             : 64   — rank das matrizes de decomposição (A e B)
    lora_alpha    : 16   — fator de escala α; o update é escalado por α/r = 0.25
    lora_dropout  : 0.1  — dropout aplicado às camadas LoRA (regularização)
    bias          : "none" — não treina biases das camadas injetadas
    task_type     : CAUSAL_LM — tarefa de geração de linguagem causal (GPT-style)
    target_modules: camadas alvo onde as matrizes LoRA serão injetadas
                    (projeções de atenção Q, K, V e projeção de saída)

    Retorna
    -------
    lora_config : LoraConfig
    """
    lora_config = LoraConfig(
        r             = 64,
        lora_alpha    = 16,
        lora_dropout  = 0.1,
        bias          = "none",
        task_type     = TaskType.CAUSAL_LM,
        target_modules = [
            "q_proj",   # projeção Query da atenção
            "k_proj",   # projeção Key da atenção
            "v_proj",   # projeção Value da atenção
            "o_proj",   # projeção de saída da atenção
        ],
    )
    return lora_config


def aplicar_lora(model, lora_config):
    """
    Aplica as adaptações LoRA ao modelo base usando get_peft_model.

    Congela todos os parâmetros originais do modelo e adiciona os adaptadores
    LoRA como parâmetros treináveis nas camadas alvo especificadas.

    Parâmetros
    ----------
    model       : AutoModelForCausalLM  — modelo base quantizado (do Passo 2)
    lora_config : LoraConfig

    Retorna
    -------
    model : PeftModel  — modelo com adaptadores LoRA injetados
    """
    model = get_peft_model(model, lora_config)
    return model


def contar_parametros(model):
    """
    Conta e exibe os parâmetros treináveis vs totais do modelo.

    Parâmetros
    ----------
    model : PeftModel

    Retorna
    -------
    tuple(int, int, float) : (treináveis, totais, percentual)
    """
    total     = sum(p.numel() for p in model.parameters())
    treinavel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100.0 * treinavel / total if total > 0 else 0.0
    return treinavel, total, pct


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("PASSO 3 — Arquitetura do LoRA")
    print("=" * 65)

    lora_config = criar_lora_config()

    print("\nLoraConfig instanciado:")
    print(f"  r (rank)       = {lora_config.r}")
    print(f"  lora_alpha     = {lora_config.lora_alpha}")
    print(f"  lora_dropout   = {lora_config.lora_dropout}")
    print(f"  bias           = {lora_config.bias}")
    print(f"  task_type      = {lora_config.task_type}")
    print(f"  target_modules = {lora_config.target_modules}")

    print(f"\nFator de escala efetivo (alpha/r) = {lora_config.lora_alpha}/{lora_config.r}"
          f" = {lora_config.lora_alpha/lora_config.r:.4f}")

    print("\nEquação LoRA:")
    print("  W' = W_congelado + (alpha/r) * B @ A")
    print("  onde A ∈ ℝ^(r×k) e B ∈ ℝ^(d×r), com r << min(d, k)")

    print("\nNota: Para aplicar ao modelo, execute após o Passo 2:")
    print("  model = aplicar_lora(model_base, lora_config)")
    print("  model.print_trainable_parameters()")

    print("\nEfeito esperado nos parâmetros (Llama 2 7B):")
    print("  Parâmetros totais     : ~6.700.000.000")
    print("  Parâmetros treináveis : ~     33.554.432  (≈ 0.5%)")
    print("  Redução               : ~99.5% dos parâmetros congelados")

    print("\n✓ LoraConfig configurado corretamente.")
    print("=" * 65)

    return lora_config


if __name__ == "__main__":
    demo()
