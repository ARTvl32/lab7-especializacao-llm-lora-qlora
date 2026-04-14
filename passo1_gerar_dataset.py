"""
Laboratório 7 — Passo 1: Engenharia de Dados Sintéticos (Dataset Generation)
=============================================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Utiliza a API da OpenAI (GPT-3.5-turbo) para gerar um dataset sintético de
instruções no domínio de Engenharia de Software.

O script gera pelo menos 50 pares de prompt (pergunta/instrução) e response
(resposta esperada), divide os dados em 90% treino e 10% teste, e salva
os resultados no formato .jsonl.

Domínio escolhido: Engenharia de Software
(boas práticas, padrões de projeto, arquitetura, testes, refatoração)

Nota sobre IA
-------------
Os pares de instrução/resposta são gerados via API da OpenAI (GPT-3.5-turbo).
O script de chamada à API, o prompt de sistema e a lógica de parsing foram
gerados/complementados com IA, revisados por Arthur.

Dependências
------------
    pip install openai
    Variável de ambiente: OPENAI_API_KEY
"""

import os
import json
import random
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

TOTAL_PAIRS    = 50
TRAIN_RATIO    = 0.90
OUTPUT_DIR     = "data"
TRAIN_FILE     = os.path.join(OUTPUT_DIR, "dataset_treino.jsonl")
TEST_FILE      = os.path.join(OUTPUT_DIR, "dataset_teste.jsonl")
DOMAIN         = "Engenharia de Software"
MODEL          = "gpt-3.5-turbo"

# Tópicos do domínio — garante diversidade no dataset gerado
TOPICS = [
    "princípios SOLID",
    "padrões de projeto (Design Patterns)",
    "arquitetura limpa (Clean Architecture)",
    "testes unitários e TDD",
    "refatoração de código",
    "controle de versão com Git",
    "integração contínua e entrega contínua (CI/CD)",
    "boas práticas de revisão de código",
    "dívida técnica",
    "microsserviços vs monolito",
]


# ---------------------------------------------------------------------------
# Passo 1.1 — Gerar pares via API da OpenAI
# ---------------------------------------------------------------------------

def gerar_par(client, topico):
    """
    Gera um único par {prompt, response} sobre o tópico fornecido.

    Parâmetros
    ----------
    client : OpenAI  — cliente autenticado
    topico : str     — tópico do domínio

    Retorna
    -------
    par : dict {"prompt": str, "response": str}  ou None em caso de erro
    """
    system_prompt = (
        "Você é um instrutor especializado em Engenharia de Software. "
        "Gere exatamente 1 par de instrução e resposta no formato JSON com "
        "as chaves 'prompt' e 'response'. "
        "O 'prompt' deve ser uma pergunta prática de um estudante de "
        "Engenharia de Software. A 'response' deve ser uma resposta técnica "
        "clara e didática com no mínimo 3 frases. "
        "Retorne APENAS o JSON, sem texto adicional, sem markdown."
    )

    user_prompt = f"Gere um par de instrução/resposta sobre o tópico: {topico}"

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=400,
        )

        content = completion.choices[0].message.content.strip()
        par     = json.loads(content)

        # Validar que as chaves esperadas existem
        assert "prompt"   in par, "Chave 'prompt' ausente"
        assert "response" in par, "Chave 'response' ausente"

        return par

    except Exception as e:
        print(f"  ⚠ Erro ao gerar par para '{topico}': {e}")
        return None


def gerar_dataset(total=TOTAL_PAIRS):
    """
    Gera `total` pares de instrução/resposta distribuídos entre os tópicos.

    Parâmetros
    ----------
    total : int  — número total de pares a gerar (mínimo 50)

    Retorna
    -------
    pares : list[dict]  — lista de pares {"prompt": ..., "response": ...}
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variável de ambiente OPENAI_API_KEY não definida.\n"
            "Execute: export OPENAI_API_KEY='sua-chave-aqui'"
        )

    client = OpenAI(api_key=api_key)

    print(f"Gerando {total} pares de instrução/resposta via GPT-3.5-turbo...")
    print(f"Domínio: {DOMAIN}\n")

    pares = []
    tentativas = 0

    while len(pares) < total:
        tentativas += 1
        topico = TOPICS[tentativas % len(TOPICS)]

        print(f"  [{len(pares)+1:2d}/{total}] Tópico: '{topico}'", end=" ")
        par = gerar_par(client, topico)

        if par:
            pares.append(par)
            print("✓")
        else:
            print("✗ (pulando)")

        if tentativas > total * 2:
            print("⚠ Muitas falhas consecutivas. Encerrando geração.")
            break

    print(f"\n✓ {len(pares)} pares gerados com sucesso.")
    return pares


# ---------------------------------------------------------------------------
# Passo 1.2 — Dividir e salvar em .jsonl
# ---------------------------------------------------------------------------

def dividir_e_salvar(pares, train_ratio=TRAIN_RATIO):
    """
    Embaralha os pares, divide em treino/teste e salva em arquivos .jsonl.

    Parâmetros
    ----------
    pares       : list[dict]  — pares gerados
    train_ratio : float       — fração de treino (ex: 0.90 = 90%)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(42)
    random.shuffle(pares)

    n_treino = int(len(pares) * train_ratio)
    treino   = pares[:n_treino]
    teste    = pares[n_treino:]

    def salvar_jsonl(dados, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            for item in dados:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✓ Salvo: {caminho} ({len(dados)} pares)")

    print(f"\nDividindo: {len(treino)} treino / {len(teste)} teste")
    salvar_jsonl(treino, TRAIN_FILE)
    salvar_jsonl(teste,  TEST_FILE)


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("PASSO 1 — Engenharia de Dados Sintéticos (Dataset Generation)")
    print("=" * 65)

    pares = gerar_dataset(total=TOTAL_PAIRS)
    dividir_e_salvar(pares, train_ratio=TRAIN_RATIO)

    # Exibir exemplos
    print(f"\nExemplos do dataset gerado:")
    for i, par in enumerate(pares[:3]):
        print(f"\n  [{i+1}]")
        print(f"  prompt   : {par['prompt']}")
        print(f"  response : {par['response'][:120]}...")

    print(f"\n✓ Dataset salvo em '{OUTPUT_DIR}/'")
    print(f"  Treino : {TRAIN_FILE}")
    print(f"  Teste  : {TEST_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    demo()
