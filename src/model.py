"""
model.py
--------
Carregamento do modelo base com quantização 4-bit (QLoRA) e
aplicação das configurações LoRA ou DoRA via PEFT.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)



# ---------------------------------------------------------------------------
# Configuração de quantização
# ---------------------------------------------------------------------------



def criar_bnb_config(compute_dtype: torch.dtype = torch.bfloat16) -> BitsAndBytesConfig:
    """
    Cria a configuração de quantização NF4 (double quant) para QLoRA.

    Args:
        compute_dtype: Dtype das operações (bfloat16 recomendado para Ampere+).

    Returns:
        BitsAndBytesConfig pronta para uso.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # Double quantization do QLoRA
    )



# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------



def carregar_tokenizador(model_id: str) -> AutoTokenizer:
    """
    Carrega e configura o tokenizador.

    Args:
        model_id: Nome ou caminho do modelo no Hugging Face Hub.

    Returns:
        Tokenizador configurado.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Llama não tem pad_token por padrão
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"  # Evita warnings no SFTTrainer
    print(f"[model] Tokenizador carregado: {model_id}")
    return tokenizer




# ---------------------------------------------------------------------------
# Modelo base
# ---------------------------------------------------------------------------



def carregar_modelo_base(
    model_id: str,
    bnb_config: BitsAndBytesConfig | None = None,
) -> AutoModelForCausalLM:
    """
    Carrega o modelo base, com ou sem quantização.

    Args:
        model_id: Nome ou caminho do modelo.
        bnb_config: Configuração BitsAndBytes; se None, carrega em fp16/bf16.

    Returns:
        Modelo carregado na(s) GPU(s) disponível(is).
    """
    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False           # Necessário para gradient checkpointing
    model.config.pretraining_tp = 1          # Evita inconsistências no Llama

    print(f"[model] Modelo base carregado: {model_id}")
    return model




# ---------------------------------------------------------------------------
# Configurações PEFT
# ---------------------------------------------------------------------------



# Módulos alvo para família Llama (atenção + MLP)
LLAMA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]



def criar_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] = LLAMA_TARGET_MODULES,
    use_dora: bool = False,
) -> LoraConfig:
    """
    Cria a configuração LoRA ou DoRA.

    Args:
        r: Rank das matrizes de baixo posto.
        lora_alpha: Fator de escala (geralmente 2 * r).
        lora_dropout: Dropout aplicado sobre os adaptadores.
        target_modules: Lista de módulos de atenção/MLP alvo.
        use_dora: Se True, ativa Weight-Decomposed LoRA (DoRA).

    Returns:
        LoraConfig configurada.
    """
    nome_tecnica = "DoRA" if use_dora else "LoRA"
    print(f"[model] Configurando {nome_tecnica} | rank={r} | alpha={lora_alpha}")

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,          # <<< A única diferença entre LoRA e DoRA
    )



# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------



def preparar_modelo_peft(
    model_id: str,
    use_dora: bool = False,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_qlora: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Pipeline completo: carrega tokenizador + modelo quantizado + adaptador PEFT.

    Args:
        model_id: ID do modelo base no Hub.
        use_dora: True para DoRA, False para LoRA.
        r: Rank dos adaptadores.
        lora_alpha: Alpha dos adaptadores.
        lora_dropout: Dropout dos adaptadores.
        use_qlora: Se True, aplica quantização 4-bit.

    Returns:
        Tupla (model_peft, tokenizer) prontos para treino.
    """
    tokenizer = carregar_tokenizador(model_id)

    bnb_config = criar_bnb_config() if use_qlora else None
    model = carregar_modelo_base(model_id, bnb_config)

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = criar_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def carregar_modelo_salvo(
    model_id: str,
    checkpoint_dir: str,
) -> tuple[PeftModel, AutoTokenizer]:
    """
    Carrega um modelo PEFT já treinado a partir de um checkpoint.

    Args:
        model_id: ID do modelo base original.
        checkpoint_dir: Caminho para o diretório salvo pelo Trainer.

    Returns:
        Tupla (model_peft, tokenizer) prontos para inferência.
    """
    tokenizer = carregar_tokenizador(model_id)
    bnb_config = criar_bnb_config()
    base_model = carregar_modelo_base(model_id, bnb_config)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    print(f"[model] Checkpoint carregado: {checkpoint_dir}")
    return model, tokenizer



# ---------------------------------------------------------------------------
# Execução direta (smoke test)
# ---------------------------------------------------------------------------



if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-3.2-3B"
    

    print("=== Testando LoRA ===")
    model_lora, tok = preparar_modelo_peft(MODEL_ID, use_dora=False, r=16)

    print("\n=== Testando DoRA ===")
    model_dora, _ = preparar_modelo_peft(MODEL_ID, use_dora=True, r=16)