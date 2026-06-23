"""
evaluate.py
-----------
Avaliação dos modelos treinados no GSM8K-ptbr.

Métricas quantitativas:
  - Acurácia de resposta final (extração do padrão #### N)

Métricas qualitativas (análise do CoT):
  - Média de passos de raciocínio gerados
  - Proporção de gerações com cálculo explícito

Uso:
    # Avaliar os dois modelos e comparar
    python src/evaluate.py \
        --lora_dir resultados/lora \
        --dora_dir resultados/dora \
        --model_id meta-llama/Llama-3.2-3B \
        --n_amostras 200

    # Avaliar apenas um modelo
    python src/evaluate.py \
        --lora_dir resultados/lora \
        --model_id meta-llama/Llama-3.2-3B \
        --n_amostras 200 \
        --so_lora
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel



# ---------------------------------------------------------------------------
# Templates (espelha dataset.py sem criar dependência circular)
# ---------------------------------------------------------------------------



PROMPT_INFERENCIA = """\
### Instrução:
Resolva o seguinte problema matemático passo a passo em português.

### Problema:
{question}

### Raciocínio e Resposta:
"""

PROMPT_BASELINE_INSTRUCT = """\
### Instrução:
Resolva o problema passo a passo. Numere cada passo. Ao final, escreva a resposta no formato: #### <número>

### Problema:
{question}

### Solução:
"""

PROMPT_BASELINE_PURO = """\
Pergunta: Uma fazenda tem 3 vacas e 5 galinhas. Quantos animais há no total?
Resposta: Há 3 + 5 = <<3+5=8>>8 animais no total.
#### 8

Pergunta: Maria tem 10 maçãs e dá 3 para João. Quantas maçãs Maria tem agora?
Resposta: Maria fica com 10 - 3 = <<10-3=7>>7 maçãs.
#### 7

Pergunta: {question}
Resposta:"""


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------



@dataclass
class ResultadoExemplo:
    idx: int
    pergunta: str
    resposta_correta: str
    geracao: str
    resposta_extraida: str | None
    correto: bool
    n_passos_cot: int
    tem_calculo: bool


@dataclass
class ResultadosGerais:
    tecnica: str
    model_id: str
    checkpoint_dir: str
    n_amostras: int
    acuracia: float
    media_passos_cot: float
    proporcao_com_calculo: float
    exemplos: list[ResultadoExemplo] = field(default_factory=list)



# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------



def carregar_modelo(model_id: str, checkpoint_dir: str):
    """
    Carrega modelo base quantizado em 4-bit + adaptador PEFT salvo.
    Funciona tanto para LoRA quanto para DoRA (o checkpoint carrega
    automaticamente o use_dora correto via adapter_config.json).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[evaluate] Carregando modelo base: {model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[evaluate] Carregando adaptador: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # Melhor para geração

    return model, tokenizer



# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------



def extrair_resposta_final(texto: str) -> str | None:
    # 1. Padrão limpo: #### 42
    match = re.search(r"####\s*(-?\d[\d,.]*)", texto)
    if match:
        return match.group(1).replace(",", "").replace(".", "")

    # 2. #### seguido de texto e depois número: "#### A resposta é: 42"
    match = re.search(r"####[^#\d-]*(-?\d[\d,.]*)", texto)
    if match:
        return match.group(1).replace(",", "").replace(".", "")

    # 3. Último <<cálculo=RESULTADO>>
    matches = re.findall(r"<<[^=]+=(-?\d[\d,.]*)>>", texto)
    if matches:
        return matches[-1].replace(",", "").replace(".", "")

    return None



def contar_passos_cot(texto: str) -> int:
    """
    Conta passos de raciocínio explícitos ('Passo 1:', '1.', etc.).
    Fallback: conta frases com mais de 10 caracteres.
    """
    marcadores = re.findall(
        r"(passo\s*\d+|step\s*\d+|\n\d+[\.\)])", texto, re.IGNORECASE
    )
    if marcadores:
        return len(marcadores)
    frases = re.split(r"(?<=[.!?])\s+", texto)
    return max(1, len([f for f in frases if len(f.strip()) > 10]))



def tem_calculo_explicito(texto: str) -> bool:
    """True se houver operação matemática explícita no texto."""
    return bool(re.search(r"\d+\s*[+\-×÷*/=]\s*\d+", texto))



# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------



"""
def gerar_resposta(
    model,
    tokenizer,
    pergunta: str,
    max_new_tokens: int = 300,
) -> str:
    """"""Gera raciocínio + resposta para uma pergunta. Usa greedy decoding.""""""
    prompt = PROMPT_INFERENCIA.format(question=pergunta)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Descarta tokens do prompt, decodifica apenas o gerado
    n_prompt = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output_ids[0][n_prompt:], skip_special_tokens=True).strip()

"""

# teste
def gerar_resposta(
    model,
    tokenizer,
    pergunta: str,
    max_new_tokens: int = 300,
    prompt_template: str = PROMPT_INFERENCIA,
) -> str:
    prompt = prompt_template.format(question=pergunta)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    # Gera o raciocínio normalmente
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    n_prompt = inputs["input_ids"].shape[-1]
    texto_gerado = tokenizer.decode(output_ids[0][n_prompt:], skip_special_tokens=True).strip()

    # --- Força o número final ---
    # Se já tem #### seguido de número, retorna direto
    if re.search(r"####\s*-?\d+", texto_gerado):
        return texto_gerado

    # Senão, faz uma segunda geração com o prompt + raciocínio + "#### "
    # forçando o modelo a completar apenas com o número
    prompt_forcado = prompt + texto_gerado + "\n#### "
    inputs2 = tokenizer(
        prompt_forcado,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    ).to(model.device)

    # Tokens permitidos: apenas dígitos e espaço
    digitos = [
        tok_id
        for tok, tok_id in tokenizer.get_vocab().items()
        if re.fullmatch(r"\s*-?\d+\s*", tok)
    ]

    with torch.no_grad():
        output_ids2 = model.generate(
            **inputs2,
            max_new_tokens=10,          # número curto
            do_sample=False,
            bad_words_ids=None,
        )

    n_prompt2 = inputs2["input_ids"].shape[-1]
    numero = tokenizer.decode(output_ids2[0][n_prompt2:], skip_special_tokens=True).strip()

    # Remove qualquer texto após o número
    numero_limpo = re.search(r"-?\d+", numero)
    if numero_limpo:
        return texto_gerado + "\n#### " + numero_limpo.group(0)

    return texto_gerado

# ---------------------------------------------------------------------------
# Avaliação principal
# ---------------------------------------------------------------------------



def avaliar_modelo(
    model,
    tokenizer,
    dataset_test,
    tecnica: str,
    model_id: str,
    checkpoint_dir: str,
    n_amostras: int = 200,
    max_new_tokens: int = 300,
    prompt_template: str = PROMPT_INFERENCIA,
) -> ResultadosGerais:
    """
    Roda inferência em n_amostras e coleta métricas quantitativas e qualitativas.
    """
    amostras = dataset_test.select(range(min(n_amostras, len(dataset_test))))
    exemplos: list[ResultadoExemplo] = []
    corretos = 0

    for idx, ex in enumerate(tqdm(amostras, desc=f"[{tecnica.upper()}]")):
        geracao = gerar_resposta(model, tokenizer, ex["question"], max_new_tokens, prompt_template)

        resp_extraida = extrair_resposta_final(geracao)
        resp_correta  = extrair_resposta_final(ex["answer"])
        acerto = resp_extraida is not None and resp_extraida == resp_correta
        if acerto:
            corretos += 1

        exemplos.append(ResultadoExemplo(
            idx=idx,
            pergunta=ex["question"],
            resposta_correta=resp_correta or "N/A",
            geracao=geracao,
            resposta_extraida=resp_extraida,
            correto=acerto,
            n_passos_cot=contar_passos_cot(geracao),
            tem_calculo=tem_calculo_explicito(geracao),
        ))

    n = len(amostras)
    acuracia         = corretos / n
    media_passos     = sum(e.n_passos_cot for e in exemplos) / n
    prop_calculo     = sum(e.tem_calculo   for e in exemplos) / n

    print(f"\n[{tecnica.upper()}] Acurácia            : {acuracia:.2%}  ({corretos}/{n})")
    print(f"[{tecnica.upper()}] Média passos CoT     : {media_passos:.1f}")
    print(f"[{tecnica.upper()}] Com cálculo explícito: {prop_calculo:.1%}")

    return ResultadosGerais(
        tecnica=tecnica,
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        n_amostras=n,
        acuracia=acuracia,
        media_passos_cot=media_passos,
        proporcao_com_calculo=prop_calculo,
        exemplos=exemplos,
    )



# ---------------------------------------------------------------------------
# Comparação e persistência
# ---------------------------------------------------------------------------



def comparar(res_lora: ResultadosGerais, res_dora: ResultadosGerais) -> None:
    """Tabela comparativa lado a lado."""
    print("\n" + "=" * 56)
    print(f"{'Métrica':<36} {'LoRA':>8} {'DoRA':>8}")
    print("=" * 56)
    print(f"{'Acurácia':<36} {res_lora.acuracia:>7.2%} {res_dora.acuracia:>7.2%}")
    print(f"{'Média de passos CoT':<36} {res_lora.media_passos_cot:>8.1f} {res_dora.media_passos_cot:>8.1f}")
    print(f"{'Com cálculo explícito':<36} {res_lora.proporcao_com_calculo:>7.1%} {res_dora.proporcao_com_calculo:>7.1%}")
    print("=" * 56)

    delta = res_dora.acuracia - res_lora.acuracia
    if abs(delta) < 0.001:
        print("\n→ Empate técnico em acurácia.")
    else:
        vencedor = "DoRA" if delta > 0 else "LoRA"
        print(f"\n→ {vencedor} venceu em acurácia por {abs(delta):.2%}")

    ids_lora = {e.idx for e in res_lora.exemplos if e.correto}
    ids_dora = {e.idx for e in res_dora.exemplos if e.correto}
    print(f"→ Questões acertadas APENAS pelo LoRA : {len(ids_lora - ids_dora)}")
    print(f"→ Questões acertadas APENAS pelo DoRA : {len(ids_dora - ids_lora)}")


def salvar(resultados: ResultadosGerais, caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(asdict(resultados), f, ensure_ascii=False, indent=2)
    print(f"[evaluate] Salvo em: {caminho}")


# testeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

def carregar_modelo_base(model_id: str):
    """Carrega apenas o modelo base, sem adaptador PEFT."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[evaluate] Carregando modelo BASE (sem fine-tuning): {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação LoRA vs DoRA — GSM8K-ptbr")
    p.add_argument("--model_id",    type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--lora_dir",    type=str, default="resultados/lora")
    p.add_argument("--dora_dir",    type=str, default="resultados/dora")
    p.add_argument("--dataset",     type=str, default="Polygl0t/gsm8k-pt")
    p.add_argument("--n_amostras",  type=int, default=200)
    p.add_argument("--max_tokens",  type=int, default=300)
    p.add_argument("--rank", type=int, default=None, help="Rank para montar os diretórios automaticamente.")
    p.add_argument("--epochs", type=int, default=None, help="Quantidade de épocas usadas para o treinamento do modelo.")
    p.add_argument("--so_lora",     action="store_true", help="Avaliar apenas LoRA.")
    p.add_argument("--so_dora",     action="store_true", help="Avaliar apenas DoRA.")
    p.add_argument("--baseline", action="store_true", help="Avaliar modelo base sem fine-tuning.")
    p.add_argument("--lr_tag", type=str, default="2en4", help="Tag do lr para o diretório (ex: 2e-4). Se None, usa o valor de --lr.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Se --rank fornecido, monta os diretórios automaticamente
    if args.rank is not None:
        args.lora_dir = f"resultados/lora/r{args.rank}-ep{args.epochs}-lr{args.lr_tag}"
        args.dora_dir = f"resultados/dora/r{args.rank}-ep{args.epochs}-lr{args.lr_tag}"

    from datasets import load_dataset
    print(f"[evaluate] Carregando dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    test_split = ds["test"]

    res_lora = res_dora = None

    if args.baseline: # teste da silva
        print("\n>>> BASELINE (sem fine-tuning)")
        model, tok = carregar_modelo_base(args.model_id)
        eh_instruct = "instruct" in args.model_id.lower()
        prompt = PROMPT_BASELINE_INSTRUCT if eh_instruct else PROMPT_BASELINE_PURO
        res_base = avaliar_modelo(
            model, tok, test_split,
            tecnica="baseline",
            model_id=args.model_id,
            checkpoint_dir="none",
            n_amostras=args.n_amostras,
            max_new_tokens=args.max_tokens,
            prompt_template=prompt,
        )
        salvar(res_base, "resultados/baseline/avaliacao.json")
        exit(0)

    if not args.so_dora:
        print("\n>>> LoRA")
        model, tok = carregar_modelo(args.model_id, args.lora_dir)
        res_lora = avaliar_modelo(
            model, tok, test_split,
            tecnica="lora",
            model_id=args.model_id,
            checkpoint_dir=args.lora_dir,
            n_amostras=args.n_amostras,
            max_new_tokens=args.max_tokens,
        )
        salvar(res_lora, f"{args.lora_dir}/avaliacao.json")   # <-- era hardcoded
        del model, tok
        torch.cuda.empty_cache()

    if not args.so_lora:
        print("\n>>> DoRA")
        model, tok = carregar_modelo(args.model_id, args.dora_dir)
        res_dora = avaliar_modelo(
            model, tok, test_split,
            tecnica="dora",
            model_id=args.model_id,
            checkpoint_dir=args.dora_dir,
            n_amostras=args.n_amostras,
            max_new_tokens=args.max_tokens,
        )
        salvar(res_dora, f"{args.dora_dir}/avaliacao.json")   # <-- era hardcoded

    if res_lora and res_dora:
        comparar(res_lora, res_dora)