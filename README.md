# Fine-Tuning Eficiente para Raciocínio Matemático em Português

**Um estudo comparativo entre LoRA, RSLoRA e DoRA aplicados ao Llama-3.2-3B no GSM8K-pt**

> Erique Enrique Gomes da Silva

---

## Visão Geral

Pipeline completo de fine-tuning eficiente (PEFT + QLoRA) para adaptar o modelo base `meta-llama/Llama-3.2-3B` para raciocínio matemático em português, com comparação experimental entre LoRA, RSLoRA e DoRA ao longo de mais de 20 configurações distintas rastreadas via Weights & Biases.

**Melhores resultados:**

| Configuração | Acurácia GSM8K-ptbr | Params treináveis (%)|
|---|---|---|
| RSLoRA r=8, 5ep | **36,0%** | 0.3770 |
| DoRA r=128, 5ep | **39,5%** | 5.7087 |
| LoRA r=16, 5ep | 28,5% | 0.7511 |
| Baseline (sem FT) | 11,0% | - |

---

## Estrutura do Projeto

```
projeto_lora_dora/
├── data/
│   ├── gsm8k_ptbr/          # Cache local do dataset (gerado automaticamente)
│   └── pares_pt_en.json     # Pares pt+en para data augmentation (opcional)
│
├── src/
│   ├── dataset.py           # Carregamento, formatação e splits do GSM8K-ptbr
│   ├── model.py             # Modelo base com QLoRA + configuração LoRA/DoRA
│   ├── train.py             # Loop de treino (SFTTrainer) com todos os argumentos
│   ├── evaluate.py          # Inferência, extração de resposta e métricas
│   ├── plot_loss.py         # Curvas de loss a partir dos logs do Trainer
│   └── plot_rank_efetivo.py # Análise SVD das matrizes adaptadas
│
├── resultados/
│   ├── lora/
│   ├── dora/
│   └── baseline/
├── run_experiments.sh       # Loop bash para treinar e avaliar múltiplas configs
├── run_lr_search.sh         # Varredura de learning rate
├── requirements.txt
└── README.md
```

---

## Instalação

### 1. Ambiente Python

```bash
python3.12 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

### 2. Acesso ao Llama

O modelo requer aceitação dos termos no Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login   # insira seu token HF com acesso ao Llama
```

### 3. Weights & Biases (opcional)

```bash
wandb login
```

---

## `requirements.txt`

```
torch>=2.3.0
transformers==5.8.1
peft==0.12.0
bitsandbytes==0.43.0
datasets==2.21.0
trl==0.11.4
accelerate==0.33.0
wandb
muon
sentencepiece
protobuf
scipy
```

---

## Módulos

### `dataset.py`

Carrega e formata o GSM8K-ptbr (`Polygl0t/gsm8k-pt`). O split de validação é separado do teste para evitar data leak na seleção de checkpoint.

**Funções principais:**

| Função | Descrição |
|---|---|
| `carregar_gsm8k_ptbr(fonte)` | Carrega o dataset do HF Hub; fallback para inglês se falhar |
| `preparar_dataset(dataset)` | Aplica o template de instrução em todos os splits |
| `preparar_dataset_com_validacao(dataset)` | Divide train em 90/10 (train/val), mantém test fixo |
| `preparar_dataset_merge_pt_en(pares)` | Data augmentation pt+en; split feito sobre os pares para evitar contaminação |
| `extrair_resposta_final(texto)` | Extrai o número após `#### ` no padrão GSM8K |

**Formato de prompt (treino):**
```
### Instrução:
Resolva o problema passo a passo. Numere cada passo. Ao final, escreva a resposta no formato: #### <número>

### Problema:
{question}

### Solução:
{answer}
```

**Smoke test:**
```bash
python src/dataset.py
```

---

### `model.py`

Carrega o modelo base com quantização NF4 4-bit (QLoRA) e aplica os adaptadores PEFT.

**Funções principais:**

| Função | Descrição |
|---|---|
| `criar_bnb_config()` | Configuração NF4 + double quantization + bfloat16 |
| `carregar_tokenizador(model_id)` | Tokenizador com `pad_token = eos_token` |
| `carregar_modelo_base(model_id, bnb_config)` | Modelo quantizado com `device_map="auto"` |
| `criar_lora_config(r, alpha, dropout, use_dora)` | `LoraConfig` — `use_dora=True` ativa DoRA |
| `preparar_modelo_peft(model_id, use_dora, r, ...)` | Pipeline completo: base + QLoRA + adaptador |
| `carregar_modelo_salvo(model_id, checkpoint_dir)` | Carrega `PeftModel` de um checkpoint salvo |

**Módulos alvo:**
```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

### `train.py`

Treinamento supervisionado via `SFTTrainer` (TRL). Suporta LoRA, DoRA, RSLoRA e Muon.

**Uso:**
```bash
python src/train.py --tecnica lora --rank 16 --alpha 32 --epochs 5 --lr 2e-4 --wandb
python src/train.py --tecnica dora --rank 128 --alpha 256 --epochs 5 --lr 2e-4 --wandb
```

**Todos os argumentos:**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--tecnica` | `lora` | `lora` ou `dora` |
| `--model_id` | `meta-llama/Llama-3.2-3B` | Modelo base no HF Hub |
| `--rank` | `16` | Rank r dos adaptadores |
| `--alpha` | `32` | Alpha (convenção: `alpha = 2 * rank`) |
| `--dropout` | `0.05` | Dropout dos adaptadores |
| `--epochs` | `3` | Número de épocas |
| `--batch` | `2` | Batch físico por GPU |
| `--grad_acc` | `4` | Gradient accumulation (batch efetivo = batch × grad_acc) |
| `--lr` | `2e-4` | Learning rate |
| `--lr_tag` | `2en4` | Tag do LR para nome do diretório |
| `--max_len` | `512` | Comprimento máximo de tokens |
| `--dataset` | `Polygl0t/gsm8k-pt` | Dataset no HF Hub |
| `--dataset_mode` | `padrao` | `padrao` ou `merge_pt_en` |
| `--early_stopping_patience` | `None` | Épocas sem melhora antes de parar |
| `--muon` | `False` | Usar Muon como otimizador nas matrizes 2D |
| `--wandb` | `False` | Ativar logging no W&B |

**Checkpoints salvos em:**
```
resultados/{tecnica}/r{rank}-ep{epochs}-lr{lr_tag}-b{batch}x{grad_acc}/
```

**Notas importantes:**
- `eval_dataset` usa apenas o split de **validação em português** — nunca o test set
- Checkpoint selecionado por menor `eval_loss` (`load_best_model_at_end=True`)
- Muon requer inicialização do process group: `init_process_group(backend="gloo", world_size=1)`
- RSLoRA: passar `use_rslora=True` diretamente em `model.py > criar_lora_config()`

---

### `evaluate.py`

Inferência e métricas quantitativas e qualitativas.

**Uso:**
```bash
# Avaliar par LoRA + DoRA por rank
python src/evaluate.py --rank 16 --model_id meta-llama/Llama-3.2-3B --n_amostras 200

# Avaliar apenas LoRA
python src/evaluate.py --rank 16 --so_lora

# Avaliar apenas DoRA
python src/evaluate.py --rank 16 --so_dora

# Avaliar baseline (sem fine-tuning)
python src/evaluate.py --baseline --model_id meta-llama/Llama-3.2-3B

# Passar caminhos diretos
python src/evaluate.py --lora_dir resultados/lora/r8-ep5-lr2en4-b2x4 \
                       --dora_dir resultados/dora/r8-ep5-lr2en4-b2x4
```

**Argumentos:**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--model_id` | `meta-llama/Llama-3.2-3B` | Modelo base |
| `--rank` | `None` | Monta os diretórios automaticamente |
| `--lora_dir` | `resultados/lora` | Diretório do checkpoint LoRA |
| `--dora_dir` | `resultados/dora` | Diretório do checkpoint DoRA |
| `--dataset` | `Polygl0t/gsm8k-pt` | Dataset de avaliação |
| `--n_amostras` | `200` | Amostras do test set |
| `--max_tokens` | `300` | Tokens máximos gerados |
| `--so_lora` | `False` | Avaliar apenas LoRA |
| `--so_dora` | `False` | Avaliar apenas DoRA |
| `--baseline` | `False` | Avaliar modelo base sem fine-tuning |

**Métricas coletadas:**
- `acuracia` — proporção de acertos exatos sobre `n_amostras`
- `media_passos_cot` — média de marcadores de passo no texto gerado
- `proporcao_com_calculo` — proporção de gerações com operação aritmética explícita

**Extração de resposta (hierarquia de fallback):**
1. Padrão primário: `#### N`
2. Texto intermediário: `#### A resposta é: N`
3. Último `<<expressão=N>>` no texto

Resultados salvos em `{checkpoint_dir}/avaliacao.json`.

---

### `plot_loss.py`

Plota curvas de train/eval loss a partir do `trainer_state.json` gerado pelo HuggingFace Trainer.

```bash
# Via rank e épocas (monta caminhos automaticamente)
python src/plot_loss.py --rank 32 --epochs 5

# Via caminhos diretos
python src/plot_loss.py \
    --lora_dir resultados/lora/r32-ep5-lr2en4-b2x4 \
    --dora_dir resultados/dora/r32-ep5-lr2en4-b2x4
```

Salva em `resultados/plots/loss_r{rank}_ep{epochs}.png`.

---

### `plot_rank_efetivo.py`

Análise SVD das matrizes de adaptação `ΔW = BA` de cada camada. Calcula o rank efetivo como o menor `k` tal que os `k` maiores valores singulares ao quadrado explicam 99% da energia total (`τ = 0.99`).

```bash
# Via rank e épocas
python src/plot_rank_efetivo.py --rank 32 --epochs 5

# Apenas LoRA
python src/plot_rank_efetivo.py --rank 32 --epochs 5 --so_lora

# Via caminhos diretos
python src/plot_rank_efetivo.py \
    --lora_dir resultados/lora/r32-ep5-lr2en4-b2x4 \
    --dora_dir resultados/dora/r32-ep5-lr2en4-b2x4
```

**Plots gerados:**
- `hist_rank_efetivo_{tecnica}_r{rank}_ep{epochs}.png` — histograma por técnica (bins adaptativos, máx 40)
- `hist_rank_efetivo_comparativo_r{rank}_ep{epochs}.png` — sobreposição LoRA vs DoRA

---

## Scripts de Automação

### `run_experiments.sh` — treinar e avaliar múltiplos ranks

```bash
#!/bin/bash
MODEL_ID="meta-llama/Llama-3.2-3B"
DATASET="Polygl0t/gsm8k-pt"
RANKS=(8 16 32 64)
EPOCHS=5

cd /caminho/para/projeto_lora_dora

for rank in "${RANKS[@]}"; do
    alpha=$((rank * 2))

    python src/train.py --tecnica lora --rank $rank --alpha $alpha \
        --epochs $EPOCHS --model_id $MODEL_ID --dataset $DATASET --wandb

    python src/train.py --tecnica dora --rank $rank --alpha $alpha \
        --epochs $EPOCHS --model_id $MODEL_ID --dataset $DATASET --wandb

    python src/evaluate.py --rank $rank --model_id $MODEL_ID \
        --dataset $DATASET --n_amostras 200
done
```

```bash
chmod +x run_experiments.sh
nohup ./run_experiments.sh > experimentos.log 2>&1 &
tail -f experimentos.log
```

### `run_lr_search.sh` — varredura de learning rate

```bash
#!/bin/bash
RANK=16; EPOCHS=5

for lr in 1e-3 1e-4 5e-5 1e-5; do
    lr_tag=$(echo $lr | sed 's/-/n/g')
    python src/train.py --tecnica lora --rank $RANK \
        --alpha $((RANK*2)) --epochs $EPOCHS \
        --lr $lr --lr_tag $lr_tag --wandb

    python src/evaluate.py \
        --lora_dir resultados/lora/r${RANK}-ep${EPOCHS}-lr${lr_tag}-b2x4 \
        --n_amostras 200 --so_lora
done
```

---

## Data Augmentation pt+en

Para usar data augmentation com exemplos em inglês e português do mesmo problema:

**1. Formato do arquivo `data/pares_pt_en.json`:**
```json
[
  [
    {"question": "Pergunta em pt...", "answer": "Solução em pt... #### 42"},
    {"question": "Question in en...", "answer": "Solution in en... #### 42"}
  ]
]
```

**2. Treino com augmentation:**
```bash
python src/train.py --tecnica lora --rank 32 --epochs 5 \
    --dataset_mode merge_pt_en --wandb
```

> **Nota:** O split de validação é feito sobre os **pares** (não sobre os exemplos individuais), garantindo que pt e en do mesmo problema fiquem sempre no mesmo split. A validação usa apenas exemplos em português para seleção de checkpoint — evita que o menor loss em inglês selecione um checkpoint subótimo para português.

---

## Principais Achados Experimentais

| Achado | Detalhe |
|---|---|
| **Faixa de LR segura** | `[10⁻⁴, 10⁻³]` — LR ≥ 10⁻² colapsa o modelo (acurácia 0%, geração de `!!!!!`) |
| **Saturação de rank** | Plateau em r=16–128 para LoRA; sem ganho além de r=32 |
| **RSLoRA r=8** | Melhor resultado geral: 36,0% — +15,5pp vs LoRA padrão no mesmo rank |
| **RSLoRA instável** | r ≥ 256 colapsa (scaling α/√r excessivo) — usar apenas com r ≤ 32 |
| **DoRA vs LoRA** | DoRA superou LoRA em r=128 (39,5%) ao custo de ~2,6× mais tempo |
| **Rank efetivo** | r=32 usa 94% do rank; r=1024 usa apenas 46% (6,78% dos parâmetros desperdiçados) |
| **Checkpoint ótimo** | Encontrado nas épocas 2–3; épocas extras causam overfitting no pequeno dataset |
| **Bug data aug** | Validação pt+en contamina seleção de checkpoint — usar validação somente em pt |

---

## Versões das Bibliotecas

| Biblioteca | Versão |
|---|---|
| Python | 3.12 |
| PyTorch | $\ge$ 2.3.0 |
| Transformers | 5.8.1 |
| PEFT | 0.12.0 |
| TRL | 0.11.4 |
| BitsAndBytes | 0.43.0 |
| Datasets | 2.21.0 |


---

## Referências

- Hu et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR.
- Liu et al. (2024). [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353). ICML.
- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). NeurIPS.
- Cobbe et al. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). arXiv.
- Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903). NeurIPS.
- Kalajdzievski (2023). [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732). arXiv.
- Jordan et al. (2024). [Muon: An Optimizer for Hidden Layers in Neural Networks](https://github.com/KellerJordan/Muon).