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

Tabela atual:

| Modelo      | Épocas | Tempo de Treinamento | Acurácia      | Média passos CoT | Com cálculo explícito | Data Treinamento |
|-------------|--------|----------------------|---------------|------------------|-----------------------|------------------|
|LORA r16 a32 |5       |1h46min               |39.00% (78/200)|2.1               |100.00%                |19/05/26          |
|DORA r16 a32 |5       |4h10min50s            |38.00% (76/100)|2.1               |99.5%                  |19/05/26          |


→ LoRA venceu em acurácia por 1.00%
→ Questões acertadas APENAS pelo LoRA : 19
→ Questões acertadas APENAS pelo DoRA : 17