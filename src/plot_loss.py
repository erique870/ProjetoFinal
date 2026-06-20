"""
plot_loss.py
------------
Lê os logs do Trainer salvos nos checkpoints e plota curvas de loss.

Uso:
    python src/plot_loss.py --rank 32 --epochs 5
"""

import argparse
import json
import os
import matplotlib.pyplot as plt


def carregar_logs(checkpoint_dir: str) -> list[dict]:
    """Lê o trainer_state.json salvo pelo HuggingFace Trainer."""
    caminho = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrado: {caminho}")
    with open(caminho) as f:
        state = json.load(f)
    return state["log_history"]


def extrair_curvas(logs: list[dict]) -> tuple:
    train_steps, train_loss = [], []
    eval_steps, eval_loss   = [], []

    for entry in logs:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    return train_steps, train_loss, eval_steps, eval_loss


def plotar(args):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Curvas de Loss — rank={args.rank} | epochs={args.epochs}", fontsize=13)

    cores = {"lora": "steelblue", "dora": "coral"}

    for tecnica in ["lora", "dora"]:
        ckpt_dir = f"resultados/{tecnica}/r{args.rank}-ep{args.epochs}/checkpoint-4665" # Ok, tá funcional esse pedaço
        try:
            logs = carregar_logs(ckpt_dir)
        except FileNotFoundError as e:
            print(f"[plot] Pulando {tecnica}: {e}")
            continue

        ts, tl, es, el = extrair_curvas(logs)
        cor = cores[tecnica]
        label = tecnica.upper()

        axes[0].plot(ts, tl, color=cor, label=label, linewidth=1.5)
        axes[1].plot(es, el, color=cor, label=label, linewidth=1.5, marker="o")

    for ax, titulo in zip(axes, ["Train Loss", "Eval Loss"]):
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    saida = f"resultados/plots/loss_r{args.rank}_ep{args.epochs}.png"
    os.makedirs("resultados/plots", exist_ok=True)
    plt.savefig(saida, dpi=150)
    print(f"[plot] Salvo em: {saida}")
    # plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank",   type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    plotar(p.parse_args())