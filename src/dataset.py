import re
from datasets import load_dataset, DatasetDict



# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------



PROMPT_TEMPLATE = """\
### Instrução:
Resolva o seguinte problema matemático passo a passo em português.
 
### Problema:
{question}
 
### Raciocínio e Resposta:
{answer}"""
 
PROMPT_TEMPLATE_INFERENCE = """\
### Instrução:
Resolva o seguinte problema matemático passo a passo em português.
 
### Problema:
{question}
 
### Raciocínio e Resposta:
"""
 


# ---------------------------------------------------------------------------
# Funções de carregamento
# ---------------------------------------------------------------------------



def carregar_gsm8k_ptbr(fonte: str = "Polygl0t/gsm8k-pt") -> DatasetDict:
    """
    Tenta carregar o GSM8K já traduzido para português.
    Se não encontrar, carrega o original em inglês como fallback.
 
    Args:
        fonte: Nome do dataset no Hugging Face Hub.
 
    Returns:
        DatasetDict com splits 'train' e 'test'.
    """
    try:
        dataset = load_dataset(fonte)
        print(f"[dataset] Dataset carregado: {fonte}")
    except Exception as e:
        print(f"[dataset] Falha ao carregar '{fonte}': {e}")
        print("[dataset] Usando GSM8K original (inglês) como fallback.")
        dataset = load_dataset("gsm8k", "main")
 
    print(f"[dataset] Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")
    return dataset



# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------



def formatar_exemplo(exemplo: dict, modo: str = "treino") -> dict:
    """
    Formata um exemplo do dataset no template de instrução.
 
    Args:
        exemplo: Dicionário com chaves 'question' e 'answer'.
        modo: 'treino' inclui a resposta; 'inferencia' omite.
 
    Returns:
        Dicionário com chave 'text' pronta para o tokenizador.
    """
    if modo == "treino":
        texto = PROMPT_TEMPLATE.format(
            question=exemplo["question"],
            answer=exemplo["answer"],
        )
    else:
        texto = PROMPT_TEMPLATE_INFERENCE.format(
            question=exemplo["question"],
        )
    return {"text": texto}


def preparar_dataset(dataset: DatasetDict, num_proc: int = 4) -> DatasetDict:
    """
    Aplica a formatação de instrução em todos os splits.
 
    Args:
        dataset: DatasetDict original.
        num_proc: Paralelismo no map.
 
    Returns:
        DatasetDict com coluna 'text' adicionada.
    """
    dataset_formatado = dataset.map(
        formatar_exemplo,
        fn_kwargs={"modo": "treino"},
        num_proc=num_proc,
        desc="Formatando exemplos",
    )
    return dataset_formatado



# ---------------------------------------------------------------------------
# Sanidade de extração de resposta final
# ---------------------------------------------------------------------------



def extrair_resposta_final(texto: str) -> str | None:
    """
    Extrai o número após '####' no padrão GSM8K.
 
    Exemplos:
        '...portanto o total é 42.\n#### 42'  →  '42'
        '...#### 1,234'                       →  '1234'
    """
    match = re.search(r"####\s*(-?\d[\d,]*)", texto)
    if match:
        return match.group(1).replace(",", "")
    return None



# ---------------------------------------------------------------------------
# Execução direta (smoke test)
# ---------------------------------------------------------------------------



if __name__ == "__main__":
    ds = carregar_gsm8k_ptbr()
    ds_fmt = preparar_dataset(ds)
 
    print("\n--- Exemplo formatado (treino) ---")
    print(ds_fmt["train"][0]["text"])
 
    print("\n--- Resposta extraída ---")
    resposta = extrair_resposta_final(ds_fmt["train"][0]["answer"])
    print(f"Resposta: {resposta}")