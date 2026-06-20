"""
plot_rank_efetivo.py
--------------------
Analisa o rank efetivo das matrizes LoRA/DoRA via SVD.
Plota a distribuição dos valores singulares por camada e
o rank efetivo acumulado (quantos valores singulares explicam 99% da variância).

Uso:
    python src/plot_rank_efetivo.py --rank 32 --epochs 5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def carregar_adaptador(model_id: str, checkpoint_dir: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Análise SVD
# ---------------------------------------------------------------------------

def extrair_matrizes_lora(model: PeftModel) -> dict[str, torch.Tensor]:
    """
    Extrai W = B @ A de cada camada LoRA/DoRA.
    Retorna dict {nome_camada: matriz_W}.
    """
    matrizes = {}
    for nome, modulo in model.named_modules():
        # Camadas LoRA têm lora_A e lora_B
        if hasattr(modulo, "lora_A") and hasattr(modulo, "lora_B"):
            try:
                A = modulo.lora_A["default"].weight.float().detach().cpu()
                B = modulo.lora_B["default"].weight.float().detach().cpu()
                W = B @ A  # shape: (out, in)
                matrizes[nome] = W
            except Exception:
                continue
    return matrizes


def calcular_rank_efetivo(singular_values: np.ndarray, threshold: float = 0.99) -> int:
    """
    Rank efetivo = menor k tal que os k maiores valores singulares
    explicam `threshold` da variância total (soma dos quadrados).
    """
    energia = singular_values ** 2
    energia_norm = np.cumsum(energia) / np.sum(energia)
    rank_ef = int(np.searchsorted(energia_norm, threshold)) + 1
    return min(rank_ef, len(singular_values))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plotar_rank_efetivo(args):
    fig_dir = "resultados/plots"
    os.makedirs(fig_dir, exist_ok=True)

    cores = {"lora": "steelblue", "dora": "coral"}
    resultados = {}  # tecnica -> {camada -> rank_efetivo}

    for tecnica in ["lora", "dora"]:
        ckpt_dir = f"resultados/{tecnica}/r{args.rank}-ep{args.epochs}/checkpoint-4665"
        if not os.path.exists(ckpt_dir):
            print(f"[rank_efetivo] Pulando {tecnica}: diretório não encontrado.")
            continue

        print(f"[rank_efetivo] Carregando {tecnica.upper()}...")
        model = carregar_adaptador(args.model_id, ckpt_dir)
        matrizes = extrair_matrizes_lora(model)

        if not matrizes:
            print(f"[rank_efetivo] Nenhuma matriz LoRA encontrada em {ckpt_dir}")
            continue

        ranks_efetivos = {}
        todos_singulares = []  # para o plot de distribuição

        for nome, W in matrizes.items():
            sv = torch.linalg.svdvals(W).numpy()
            todos_singulares.append(sv)
            ranks_efetivos[nome] = calcular_rank_efetivo(sv, threshold=0.99)

        resultados[tecnica] = ranks_efetivos

        # --- Plot 1: valores singulares por camada (média ± std) ---
        sv_matrix = np.array([sv[:args.rank] for sv in todos_singulares])
        media = sv_matrix.mean(axis=0)
        std   = sv_matrix.std(axis=0)
        indices = np.arange(1, args.rank + 1)

        plt.figure(figsize=(10, 4))
        plt.plot(indices, media, color=cores[tecnica], linewidth=2, label="média")
        plt.fill_between(indices, media - std, media + std,
                         alpha=0.2, color=cores[tecnica], label="±1 std")
        plt.axvline(x=np.mean(list(ranks_efetivos.values())),
                    color="red", linestyle="--", label=f"rank efetivo médio")
        plt.xlabel("Índice do valor singular")
        plt.ylabel("Magnitude")
        plt.title(f"{tecnica.upper()} r{args.rank} — Valores Singulares das Matrizes LoRA")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/singulares_{tecnica}_r{args.rank}_ep{args.epochs}.png", dpi=150)
        plt.close()

        del model
        torch.cuda.empty_cache()

    # --- Plot 2: comparação rank efetivo por camada (LoRA vs DoRA) ---
    if len(resultados) == 2:
        camadas_comuns = sorted(
            set(resultados["lora"]) & set(resultados["dora"])
        )
        # Simplifica nomes das camadas para o eixo X
        nomes_curtos = [c.split(".")[-2] + "." + c.split(".")[-1] for c in camadas_comuns]

        re_lora = [resultados["lora"][c] for c in camadas_comuns]
        re_dora = [resultados["dora"][c] for c in camadas_comuns]

        x = np.arange(len(camadas_comuns))
        largura = 0.35

        fig, ax = plt.subplots(figsize=(max(12, len(camadas_comuns) * 0.4), 5))
        ax.bar(x - largura/2, re_lora, largura, label="LoRA", color=cores["lora"], alpha=0.8)
        ax.bar(x + largura/2, re_dora, largura, label="DoRA", color=cores["dora"], alpha=0.8)
        ax.axhline(y=args.rank, color="black", linestyle="--", linewidth=1, label=f"rank nominal ({args.rank})")

        ax.set_xticks(x)
        ax.set_xticklabels(nomes_curtos, rotation=90, fontsize=7)
        ax.set_ylabel("Rank efetivo (99% da variância)")
        ax.set_title(f"Rank Efetivo por Camada — LoRA vs DoRA | r{args.rank} ep{args.epochs}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/rank_efetivo_r{args.rank}_ep{args.epochs}.png", dpi=150)
        plt.close()
        print(f"[rank_efetivo] Plots salvos em: {fig_dir}/")

        # --- Resumo no terminal ---
        print(f"\n{'='*50}")
        print(f"{'Técnica':<10} {'Rank Nominal':>14} {'Rank Efetivo Médio':>20} {'Utilização':>12}")
        print(f"{'='*50}")
        for tecnica, ranks in resultados.items():
            media_ef = np.mean(list(ranks.values()))
            utilizacao = media_ef / args.rank
            print(f"{tecnica.upper():<10} {args.rank:>14} {media_ef:>20.1f} {utilizacao:>11.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--rank",     type=int, required=True)
    p.add_argument("--epochs",   type=int, required=True)
    plotar_rank_efetivo(p.parse_args())