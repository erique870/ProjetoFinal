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

| Modelo                           | Épocas | Tempo de Treinamento | Acurácia (GSM8K-pt) | Média passos CoT | 
|----------------------------------|--------|----------------------|---------------------|------------------|
|meta-llama/Llama-3.2-3B (Baseline)|  -     |        -             |0.11                 |    -             |                  
|PEFT LoRA r = 2, a = 4            |5       |                      |                     |                  |                  
|PEFT DoRA r = 2, a = 4            |5       |                      |                     |                  |                  
|PEFT LoRA r = 4, a = 8            |5       |                      |                     |                  |                  
|PEFT DoRA r = 4, a = 8            |5       |                      |                     |                  |                  
|PEFT LoRA r = 8, a = 16           |5       |                      |                     |                  |                  
|PEFT DoRA r = 8, a = 16           |5       |                      |                     |                  |                  
|PEFT LoRA r = 16, a = 32          |5       |                      |                     |                  |                  
|PEFT DoRA r = 16, a = 32          |5       |                      |                     |                  |                  
|PEFT LoRA r = 32, a = 64          |5       |                      |                     |                  |                 
|PEFT DoRA r = 32, a = 64          |5       |                      |                     |                  |                  
|PEFT LoRA r = 64, a = 128         |5       |                      |                     |                  |                  
|PEFT DoRA r = 64, a = 128         |5       |                      |                     |                  |                 
|PEFT LoRA r = 128, a = 256        |5       |                      |                     |                  |                  
|PEFT DoRA r = 128, a = 256        |5       |                      |                     |                  |                  
|PEFT LoRA r = 256, a = 512        |5       |                      |                     |                  |                  
|PEFT DoRA r = 256, a = 512        |5       |                      |                     |                  |               


### Comentários sobre os resultados



