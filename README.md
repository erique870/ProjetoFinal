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

