# Setup IMPA Tech

---

Para usar o `cuda` crie um `venv` executando as seguintes linhas no terminal:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

```

Os drivers da placa Nvidia das máquinas do IMPATech estão desatualizados em relação a versão mais recente do `torch`. Para consertar isso, dê downgrade na biblioteca com as seguintes linha de comando no terminal:

```
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

```

Pendente: Documentar instalações de adequação do `bitsandbytes` com a versão cuda das máquinas do IMPATech.

## Primeira Bateria de Testes - Variando apenas ranks

Tabela atual (Primeira Bateria de Treinamentos):

| Modelo           | Épocas | Tempo de Treinamento | Acurácia       | Média passos CoT | Com cálculo explícito | Data Treinamento |
|------------------|--------|----------------------|----------------|------------------|-----------------------|------------------|
|LORA r16 a32 ep5  |5       |1h46min               |39.00% (78/200) |2.1               |100.00%                |19/05/26          |
|DORA r16 a32 ep5  |5       |4h10min50s            |38.00% (76/200) |2.1               |99.5%                  |19/05/26          |
|LORA r8 a16 ep3   |3       |1h31min44s            |36.00% (72/200) |2.075             |100.00%                |20/05/26          |
|DORA r8 a16 ep3   |3       |4h10min8s             |31.50% (63/200) |2.0               |99.00%                 |20/05/26          |
|LORA r16 a32 ep3  |3       |1h31min51s            |37.00% (74/200) |2.12              |99.50%                 |20/05/26          |
|DORA r16 a32 ep3  |3       |4h10min31s            |36.50% (73/200) |2.195             |99.50%                 |20/05/26          |
|LORA r32 a64 ep3  |3       |1h31min46s            |39.50% (79/200) |2.24              |98.50%                 |20/05/26          |
|DORA r32 a64 ep3  |3       |4h12min2s             |37.00% (74/200) |2.205             |99.00%                 |20/05/26          |
|LORA r64 a128 ep3 |3       |1h34min2s             |39.00% (78/200) |2.45              |99.50%                 |20/05/26          |
|DORA r64 a128 ep3 |3       |4h15min               |35.50% (71/200) |2.33              |100.00%                |20/05/26          |
|LORA r128 a256 ep3|3       |1h35min40s            |35.00% (70/200) |2.43              |100.00%                |22/05/26          |
|DORA r128 a256 ep3|3       |4h18min16s            |38.00% (76/200) |2.395             |100.00%                |22/05/26          |
|LORA r256 a512 ep3|3       |1h37min43s            |38.00% (76/200) |2.4               |99.50%                 |22/05/26          |
|DORA r256 a512 ep3|3       |4h27min42s            |32.00% (64/200) |2.295             |100.00%                |22/05/26          |


Primeiro overview: Variar apenas o rank levou a uma acurácia no range $[30, 40]$, insatisfatório. É preciso usar papers e artigos.

### Comentários sobre a primeira bateria de dados

* **Lora consistentemente melhor que o DoRA**: em quase todos os ranks o Lora demonstrou acurácia maior que o DoRA. **IMPORTANTE**: Vale investigar se DoRA precisa de de mais épocas para convergir (decomposição magnitude/direção é mais lenta)

* **Ranks maiores *não* ajudaram**: r64 e r128 não bateram r32. Isso sugere que o problema não precisa de alta capacidade paramétrica, mas sim mais passos de otimização (épocas/lr).

* **"Com cálculo explícito" está em 99%-100%**: o modelo já aprendeu a fazer contas. O gargalo possivelmente é a *lógica de múltiplos passos*, não a aritmética.

---

Agora forço o prompt enumerar os passos da solução setando `PROMPT_TEMPLATE` em `dataset.py` da seguinte maneira:

Prompt Antigo:

```PROMPT_TEMPLATE = """\
### Instrução:
Resolva o seguinte problema matemático passo a passo em português.
 
### Problema:
{question}
 
### Raciocínio e Resposta:
{answer}"""

```
Prompt Novo:

```PROMPT_TEMPLATE = """\
### Instrução:
Resolva o problema passo a passo. Numere cada passo. Ao final, escreva a resposta no formato: #### <número>

### Problema:
{question}

### Solução:
{answer}"""
```

Avalio os melhores modelos LoRA e Dora (r32a64) com 5 épocas para avaliar progresso:

Tabela atual (Primeira Bateria de Treinamentos):

| Modelo          | Épocas | Tempo de Treinamento | Acurácia       | Média passos CoT | Com cálculo explícito | Data Treinamento |
|-----------------|--------|----------------------|----------------|------------------|-----------------------|------------------|
|LORA r32 a64 ep3 |5       |                      |                |                  |                       |24/05/26          |
|DORA r32 a64 ep3 |5       |                      |                |                  |                       |24/05/26          |
