"""
train.py
--------
Treinamento supervisionado (SFT) com SFTTrainer do TRL.
Suporta LoRA e DoRA via flag --use_dora.
Salva checkpoints separados em resultados/lora/ ou resultados/dora/.

Uso:
    python src/train.py --tecnica lora --rank 16
    python src/train.py --tecnica dora --rank 16
"""

import argparse
import os
import json
import wandb
from transformers import TrainerCallback, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from muon import Muon
import torch

from dataset import carregar_gsm8k_ptbr, preparar_dataset, preparar_dataset_com_validacao, preparar_dataset_merge_pt_en
from model import preparar_modelo_peft



# ---------------------------------------------------------------------------
# Argumentos de linha de comando
# ---------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino LoRA vs DoRA no GSM8K-ptbr")

    parser.add_argument(
        "--tecnica",
        type=str,
        choices=["lora", "dora"],
        default="lora",
        help="Técnica PEFT a usar (lora ou dora).",
    )
    parser.add_argument("--model_id",  type=str,   default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--rank",      type=int,   default=16,   help="Rank r dos adaptadores.")
    parser.add_argument("--alpha",     type=int,   default=32,   help="Alpha dos adaptadores.")
    parser.add_argument("--dropout",   type=float, default=0.05)
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--batch",     type=int,   default=2,    help="Batch por GPU.")
    parser.add_argument("--grad_acc",  type=int,   default=4,    help="Acumulação de gradiente.")
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--lr_tag", type=str, default="2en4", help="Tag do lr para o diretório (ex: 2e-4). Se None, usa o valor de --lr.")
    parser.add_argument("--max_len",   type=int,   default=512,  help="Comprimento máximo de tokens.")
    parser.add_argument("--wandb",     action="store_true",      help="Ativar logging no W&B.")
    parser.add_argument("--dataset",   type=str,   default="Polygl0t/gsm8k-pt")
    parser.add_argument("--dataset_mode", type=str, choices=["padrao", "merge_pt_en"], default="padrao", help="'padrao' usa o dataset pt-br normal; 'merge_pt_en' usa data augmentation pt+en.",)
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Nº de avaliações sem melhora antes de parar. None desativa early stopping.",)
    parser.add_argument("--muon", action="store_true", help="Usar otimizador Muon nos adaptadores LoRA.")

    return parser.parse_args()



# ---------------------------------------------------------------------------
# Callback de logging customizado
# ---------------------------------------------------------------------------



class LogCallback(TrainerCallback):
    """Imprime métricas a cada passo de avaliação."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(
                f"\n[train] Epoch {state.epoch:.1f} | "
                f"eval_loss={metrics.get('eval_loss', '?'):.4f}"
            )



# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------



def treinar(args: argparse.Namespace) -> None:

        # Inicializa process group para o Muon funcionar em GPU única
    if args.muon:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            dist.init_process_group(backend="gloo")
            print("[train] Process group inicializado para Muon (single GPU).")

    use_dora = args.tecnica == "dora"
    output_dir = os.path.join("resultados", args.tecnica, f"r{args.rank}-ep{args.epochs}-lr{args.lr_tag}-b{args.batch}x{args.grad_acc}")
    run_name = f"{args.tecnica}_r{args.rank}_a{args.alpha}"

    print(f"\n{'='*60}")
    print(f" Técnica : {args.tecnica.upper()}")
    print(f" Modelo  : {args.model_id}")
    print(f" Rank    : {args.rank}  |  Alpha: {args.alpha}")
    print(f" Output  : {output_dir}")
    print(f"{'='*60}\n")

    # --- W&B ---
    if args.wandb:
        wandb.init(
            project="lora-vs-dora-math-ptbr",
            name=run_name,
            config=vars(args),
        )

    # --- Dataset ---
    
    if args.dataset_mode == "merge_pt_en":
        with open("data/pares_pt_en.json", "r", encoding="utf-8") as f:
            pares_raw = json.load(f)
        pares_pt_en = [(p[0], p[1]) for p in pares_raw]

        dataset = preparar_dataset_merge_pt_en(pares_pt_en)
    else:
        dataset_raw = carregar_gsm8k_ptbr(args.dataset)
        dataset_raw = preparar_dataset_com_validacao(dataset_raw)
        dataset = preparar_dataset(dataset_raw)

    # --- Modelo ---
    model, tokenizer = preparar_modelo_peft(
        model_id=args.model_id,
        use_dora=use_dora,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        use_qlora=True,
    )

    # --- Configuração de treino ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,

        # Épocas e batch
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,

        # Otimizador
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,             # warmup_ratio=0.05,
        weight_decay=0.01,
        optim="paged_adamw_32bit",   # Otimizado para QLoRA

        # Precisão
        bf16=True,                   # Use fp16=True se GPU não suportar bf16
        fp16=False,

        # Gradient checkpointing (economiza VRAM)
        gradient_checkpointing=True,

        # Sequência
        max_length=args.max_len,
        dataset_text_field="text",
        packing=False,               # True = agrupa sequências curtas (mais eficiente)

        # Logging e salvamento
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs, # Guarda um checkpoint por epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,   # <-- adiciona isso explicitamente (loss menor é melhor)

        # W&B
        report_to="wandb" if args.wandb else "none",
    )

    # --- Trainer ---
    if args.muon:
        lora_matrix_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]

        # Log para confirmar que encontrou parâmetros
        print(f"[train] Muon: {len(lora_matrix_params)} tensores 2D treináveis")

        optimizer = Muon(lora_matrix_params, lr=args.lr, momentum=0.95)
    else:
        optimizer = None

    callbacks=[LogCallback()]
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=callbacks,
        optimizers=(optimizer, None) if args.muon else (None, None),
    )

    # --- Treino ---
    print("[train] Iniciando treinamento...")
    trainer.train()

    # --- Salvamento final ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[train] Modelo salvo em: {output_dir}")

    if args.wandb:
        wandb.finish()



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    treinar(args)