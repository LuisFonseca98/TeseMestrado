# Instalação e preparação do projeto FAS noutra máquina

Este guia permite preparar o projeto **Face Anti-Spoofing (FAS)** num novo computador Windows, incluindo:

- criação do ambiente virtual Python;
- configuração no Visual Studio Code;
- criação/seleção do kernel Jupyter;
- instalação das dependências;
- instalação/download dos datasets;
- estrutura recomendada do projeto;
- validação da pipeline atual.

> Estado atual do projeto: a implementação experimental usa apenas **imagens RGB** dos datasets **NUAA, LCC-FASD e CASIA-FASD**.  
> AxonData é mantido para análise/estado da arte, mas não entra atualmente no treino.

---

## 1. Pré-requisitos

Instalar previamente:

- Python 3.x
- Git
- Visual Studio Code
- Extensão **Python** no VS Code
- Extensão **Jupyter** no VS Code

Opcional, mas recomendado para treino:

- GPU NVIDIA com drivers atualizados

---

## 2. Estrutura recomendada do projeto

Criar/copiar o projeto com esta estrutura:

```text
TESE/
│
├── .venv/
│
├── Datasets/
│   ├── NUAA.zip
│   ├── LCC_FASD/
│   ├── CASIA_FASD/
│   └── AxonData_FAS/              # opcional nesta fase
│
├── Pipeline/
│   ├── 01_build_image_manifest.py
│   ├── 02_build_cross_domain_splits.py
│   ├── 03_fas_dataset.py
│   ├── fas_dataset.py
│   └── 04_test_dataloader.py
│
├── Notebooks/
│
├── reports/
│   ├── fas_image_manifest.csv
│   └── experiments/
│       ├── target_nuaa.csv
│       ├── target_lcc_fasd.csv
│       └── target_casia_fasd.csv
│
└── ResumosArtigos/
```

Os scripts procuram automaticamente a raiz do projeto através da pasta `Datasets`, portanto não devem depender de um caminho absoluto específico.

---

## 3. Criar o ambiente virtual

Abrir PowerShell na pasta `TESE`.

Exemplo:

```powershell
cd "C:\CAMINHO\PARA\TESE"
```

Criar o ambiente:

```powershell
python -m venv .venv
```

Se o Windows bloquear a ativação de scripts no PowerShell, executar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Ativar o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

O terminal deverá passar a mostrar algo semelhante a:

```text
(.venv) PS C:\...\TESE>
```

Confirmar o Python usado:

```powershell
python --version
where.exe python
```

O primeiro caminho devolvido deve apontar para:

```text
TESE\.venv\Scripts\python.exe
```

---

## 4. Atualizar pip

```powershell
python -m pip install --upgrade pip
```

---

## 5. Instalar dependências do projeto

Instalar as bibliotecas utilizadas na análise dos datasets, notebooks e pipeline:

```powershell
python -m pip install jupyter ipykernel numpy pandas pillow opencv-python scikit-learn kagglehub huggingface_hub
```

Instalar PyTorch e Torchvision:

```powershell
python -m pip install torch torchvision
```

Depois confirmar:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Se `CUDA: False` numa máquina com GPU NVIDIA

Não alterar o resto do projeto.

Usar o instalador recomendado pelo site oficial do PyTorch para a versão atual de Windows/CUDA e voltar a executar o comando de verificação.

O projeto também funciona em CPU, mas o treino será significativamente mais lento.

---

## 6. Configurar o VS Code

Abrir o projeto:

```powershell
code .
```

No VS Code:

1. `Ctrl + Shift + P`
2. Escolher **Python: Select Interpreter**
3. Selecionar:

```text
TESE\.venv\Scripts\python.exe
```

Se não aparecer automaticamente:

1. `Python: Select Interpreter`
2. `Enter interpreter path`
3. Selecionar manualmente:

```text
<PASTA_TESE>\.venv\Scripts\python.exe
```

---

## 7. Criar o kernel Jupyter

Com `.venv` ativo:

```powershell
python -m ipykernel install --user --name tese --display-name "TESE (.venv)"
```

Depois, num notebook `.ipynb` no VS Code:

1. clicar em **Select Kernel**;
2. escolher **Python Environments** ou **Jupyter Kernel**;
3. selecionar:

```text
TESE (.venv)
```

Para confirmar dentro de um notebook:

```python
import sys
print(sys.executable)
```

Deve apontar para:

```text
...\TESE\.venv\Scripts\python.exe
```

---

# 8. Instalar os datasets

## 8.1 NUAA

Dataset utilizado:

```text
NUAA Photograph Imposter Database
versão Detectedface
```

Na página oficial do NUAA, descarregar a versão com as faces já detetadas:

```text
a format with face images output by a face detector
```

Guardar o ZIP dentro de:

```text
TESE\Datasets\
```

Pode chamar-se, por exemplo:

```text
NUAA.zip
```

O script atual não depende do nome exato do ZIP: identifica-o pelos ficheiros internos do protocolo oficial.

O ZIP deve conter os ficheiros de split:

```text
client_train_face.txt
client_test_face.txt
imposter_train_face.txt
imposter_test_face.txt
```

Não é necessário extrair o ZIP.

---

## 8.2 LCC-FASD

Instalar `kagglehub` caso ainda não exista:

```powershell
python -m pip install kagglehub
```

Executar a partir da raiz `TESE`:

```powershell
python -c "import kagglehub; print(kagglehub.dataset_download('faber24/lcc-fasd', output_dir=r'.\Datasets\LCC_FASD'))"
```

No fim deverão existir as três divisões:

```text
LCC_FASD_training
LCC_FASD_development
LCC_FASD_evaluation
```

O protocolo atual usa:

```text
training    -> train
development -> validation
evaluation  -> test quando LCC é target
```

---

## 8.3 CASIA-FASD

Executar:

```powershell
python -c "import kagglehub; print(kagglehub.dataset_download('minhnh2107/casiafasd', output_dir=r'.\Datasets\CASIA_FASD'))"
```

A versão utilizada no projeto já contém frames pré-processados.

A pipeline atual utiliza apenas:

```text
train_img/.../color
test_img/.../color
```

Não utiliza:

```text
depth
```

Isto é intencional porque NUAA e LCC-FASD não possuem a mesma modalidade.

---

## 8.4 AxonData — opcional nesta fase

AxonData não entra atualmente no treino image-only, mas pode ser descarregado para análise do dataset e estado da arte.

Instalar:

```powershell
python -m pip install huggingface_hub
```

Download:

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='AxonData/face-anti-spoofing-dataset', repo_type='dataset', local_dir=r'.\Datasets\AxonData_FAS', ignore_patterns=['3D_paper_mask /*']))"
```

A razão para não o usar no protocolo atual é que a amostra pública analisada contém sobretudo spoof em vídeo, enquanto a implementação atual foi deliberadamente restringida a imagens.

---

# 9. Validar os datasets e criar o manifest

Com o ambiente ativo e a partir da raiz `TESE`:

```powershell
python .\Pipeline\01_build_image_manifest.py
```

Contagens esperadas:

```text
NUAA        12 614
LCC-FASD    18 827
CASIA-FASD   4 063 RGB
-------------------
TOTAL       35 504 imagens
```

Output principal:

```text
reports\fas_image_manifest.csv
```

O manifest cria uma representação comum dos três datasets.

Inclui informação como:

```text
sample_id
dataset
domain
domain_id
native_split
label
class_id
subject_id
attack_type
quality
filepath / archive
```

Labels:

```text
0 = bona_fide
1 = spoof
```

---

# 10. Criar os três protocolos cross-domain

Executar:

```powershell
python .\Pipeline\02_build_cross_domain_splits.py
```

São criadas estas experiências:

```text
NUAA + LCC-FASD   -> CASIA-FASD
NUAA + CASIA-FASD -> LCC-FASD
LCC-FASD + CASIA  -> NUAA
```

Outputs:

```text
reports\experiments\target_casia_fasd.csv
reports\experiments\target_lcc_fasd.csv
reports\experiments\target_nuaa.csv
```

Regra fundamental:

```text
UNSEEN TARGET = TEST ONLY
```

O target nunca pode ser utilizado para:

- treino;
- validation;
- escolha do melhor checkpoint;
- escolha de hiperparâmetros;
- escolha de threshold.

---

# 11. Validar o DataLoader

Testar primeiro CASIA como target:

```powershell
python .\Pipeline\04_test_dataloader.py --target casia_fasd
```

Depois:

```powershell
python .\Pipeline\04_test_dataloader.py --target lcc_fasd
python .\Pipeline\04_test_dataloader.py --target nuaa
```

Com `batch_size = 16`, os batches de treino devem apresentar aproximadamente:

```text
class_id={0: 8, 1: 8}
```

e uma distribuição igual entre os dois source domains.

Internamente, num experimento com dois source domains, o sampler tenta produzir:

```text
4 source A + bona fide
4 source A + spoof
4 source B + bona fide
4 source B + spoof
```

Validation e test mantêm a distribuição natural dos datasets.

---

# 12. Valores atualmente fixados

```text
IMAGE_SIZE = 256 x 256
RandomHorizontalFlip p = 0.5
Split seed = 42
Validation subjects = 20% (NUAA e CASIA)
Batch size de validação inicial = 16
```

Importante:

```text
p = 0.5
```

é apenas a probabilidade de aplicar horizontal flip durante data augmentation.

Ainda **não existe um classification threshold**.

Quando existir, qualquer threshold otimizado terá de ser escolhido exclusivamente com **source validation data**, nunca com o unseen target test.

---

# 13. Estado a partir do qual continuar

Quando os três testes do DataLoader terminarem com:

```text
✓ CSV lido
✓ train/val/test carregados
✓ imagens convertidas para RGB
✓ tensors [B, 3, 256, 256]
✓ sampler train equilibrado por domain + class
✓ validation/test mantêm distribuição natural
```

a máquina está preparada para a próxima fase:

```text
ResNet18 baseline
    -> treino nos source domains
    -> source validation
    -> melhor checkpoint
    -> unseen target test
    -> métricas
```

Posteriormente:

```text
ResNet18 baseline
        vs.
SA-FAS = classification + SupCon + PG-IRM
```

usando exatamente os mesmos datasets, splits e preprocessing.
