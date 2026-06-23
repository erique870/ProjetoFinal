import re
from datasets import load_dataset, DatasetDict, Dataset
import random



# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------



PROMPT_TEMPLATE = """\
### Instrução:
Resolva o problema passo a passo. Numere cada passo. Ao final, escreva a resposta no formato: #### <número>

### Problema:
{question}

### Solução:
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


def preparar_dataset_com_validacao(dataset: DatasetDict, seed: int = 42) -> DatasetDict:
    """
    Divide train em train/val (90/10), mantém test intocado.
    Usar para evitar overfitting na seleção de checkpoint via eval_dataset.
    """
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=seed)
    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": dataset["test"],
    })



def preparar_dataset_merge_pt_en(
    pares_pt_en: list[tuple[dict, dict]],
    seed: int = 42,
) -> DatasetDict:
    """
    Constrói dataset de data augmentation a partir de tuplas (exemplo_pt, exemplo_en)
    do MESMO problema.

    O split train/val é feito sobre os PARES, garantindo que pt e en do mesmo
    problema fiquem sempre no mesmo split. Depois do split, os pares são
    desfeitos e os exemplos pt+en se misturam livremente dentro de cada split.

    Args:
        pares_pt_en: Lista de tuplas (dict_pt, dict_en), cada dict com
                     chaves 'question' e 'answer'.
        seed: Seed para shuffle e split.

    Returns:
        DatasetDict com 'train', 'validation' (pt+en misturados) e
        'test' (GSM8K-ptbr original, para métrica comparável).
    """

    rng = random.Random(seed)
    indices = list(range(len(pares_pt_en)))
    rng.shuffle(indices)

    n_val = int(len(indices) * 0.1)
    indices_val = set(indices[:n_val])
    indices_train = set(indices[n_val:])

    def _formatar_par(idx: int) -> list[dict]:
        ex_pt, ex_en = pares_pt_en[idx]
        registros = []
        for ex in (ex_pt, ex_en):
            fmt = formatar_exemplo(ex, modo="treino")
            registros.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "text": fmt["text"],
            })
        return registros

    registros_train = []
    for idx in indices_train:
        registros_train.extend(_formatar_par(idx))

    registros_val = []
    for idx in indices_val:
        registros_val.extend(_formatar_par(idx))

    ds_train = Dataset.from_list(registros_train).shuffle(seed=seed)
    ds_val   = Dataset.from_list(registros_val).shuffle(seed=seed)

    ds_ptbr_original = carregar_gsm8k_ptbr()

    print(f"[dataset] Merge pt/en: {len(indices_train)} pares treino "
          f"({len(registros_train)} exemplos) | "
          f"{len(indices_val)} pares val ({len(registros_val)} exemplos)")

    return DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_ptbr_original["test"],
    })

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