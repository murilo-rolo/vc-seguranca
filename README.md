# Detecção de Situações de Risco em Vídeos de Sistemas de Vigilância

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](LICENSE)

Sistema avançado de detecção de violência em vídeos de segurança (CCTV) utilizando Deep Learning e Visão Computacional. Este projeto implementa uma **arquitetura multimodal** que combina múltiplas fontes de informação (vídeo, pose corporal e emoção facial) para detecção precisa de situações de risco e violência.

## Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Datasets](#datasets)
- [Arquitetura dos Modelos](#arquitetura-dos-modelos)
- [Instalação](#instalação)
- [Uso](#uso)
  - [Pré-processamento](#1-pré-processamento)
  - [Treinamento](#2-treinamento)
  - [Avaliação](#3-avaliação)
  - [Inferência em Tempo Real](#5-inferência-em-tempo-real)
- [Pipeline Completo de Execução](#pipeline-completo-de-execução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos de Sistema](#requisitos-de-sistema)
- [Troubleshooting](#troubleshooting)
- [Referências](#referências)

## Visão Geral

Este projeto está sendo desenvolvido como parte de um projeto de pesquisa em Visão Computacional. O sistema é capaz de analisar vídeos de câmeras de segurança e identificar automaticamente situações de violência, auxiliando em sistemas de monitoramento automatizado.

### Aplicações

- Monitoramento automatizado de segurança
- Alertas em tempo real para situações de risco
- Análise de grandes volumes de vídeo
- Redução de falsos positivos em sistemas de segurança

## Características

- **Arquitetura Multimodal**: Combina features de vídeo, pose corporal e emoção facial para detecção robusta
- **Múltiplos Modelos**: 
  - CNN 3D (R3D, R(2+1)D, MC3) para action recognition — backbone de vídeo **padrão** do multimodal
  - ResNet-LSTM para análise temporal de vídeo (alternativo)
  - EmotionNet (DeiT-Small Vision Transformer) para classificação de emoções faciais
  - MultimodalRiskDetector para fusão de múltiplas modalidades
- **Estratégias de Fusão**: Early Fusion, Late Fusion e Attention-based Fusion
- **Transfer Learning**: Utiliza pesos pré-treinados (ImageNet, Kinetics400)
- **Pipeline Completo**: Pré-processamento, extração de features, treinamento, avaliação e inferência em tempo real
- **Detecção em Tempo Real**: Suporte para webcam e streams RTSP
- **Sistema de Alertas**: Threshold configurável e detecção de janelas consecutivas
- **Altamente Configurável**: Parâmetros ajustáveis para diferentes cenários
- **Otimizado para Recursos Limitados**: Suporta treinamento em CPUs e GPUs, mixed precision (AMP) e gradient clipping
- **Métricas Detalhadas**: Gera relatórios completos de avaliação
- **Pipeline de Avaliação Experimental**: Métricas, robustez a distorções, performance (FPS/latência) e análise de limitações
- **Treinamento Robusto do EmotionNet**: Focal Loss, WeightedRandomSampler, warmup + cosine annealing, early stopping e resume de treino
- **Download Automático de Datasets**: `download_datasets.py` baixa RWF-2000 e AffectNet via Kaggle API
- **Código modular**: Funções de treino/validação centralizadas em `src/training/utils.py`
- **Gerenciamento Centralizado de Caminhos**: `src/paths.py` detecta automaticamente o ambiente (local ou Google Colab) e configura todos os diretórios do projeto
- **Organização de Modelos por Pasta**: Pesos e experimentos organizados em `models/<modelo>/weights/` e `models/<modelo>/experiments/`

## Datasets

O projeto utiliza múltiplos datasets para diferentes propósitos:

### Dataset Principal: RWF-2000

**RWF-2000** (Real-World Fighting Dataset) - Dataset principal para detecção de violência:

- **2.000 vídeos** rotulados como `violent` (Fight) e `non_violent` (NonFight)
- Vídeos gravados por câmeras de segurança (CCTV) em cenários reais
- Divisão: conjunto de treino e validação
- Disponível em repositórios públicos (ex.: [Kaggle](https://www.kaggle.com/datasets/vulamnguyen/rwf2000/data))

**Estrutura esperada:**
```
dataset/RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

### Dataset de Emoção: Balanced-AffectNet

**Balanced-AffectNet** - Usado para treinar o modelo de reconhecimento de emoções faciais:

- **8 classes de emoção**: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise, Contempt
- **41.008 imagens** balanceadas de faces (sem desbalanceamento entre classes)
- Imagens em RGB 75×75 PNG, organizadas em pastas por classe
- Disponível em: [Balanced-AffectNet (Kaggle)](https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet)

**Estrutura esperada:**
```
dataset/balanced-affectnet/
├── train/<Classe>/*.png
├── val/<Classe>/*.png
└── test/<Classe>/*.png
```

> **Nota**: As classes são derivadas das pastas reais (ordenadas) e os labels são o índice nessa lista. O dataset balanceado não possui `labels.csv`; o formato legado `dataset/AffectNet` (com `labels.csv` e pastas `Train/`/`Test/`) continua suportado como fallback. O modelo é salvo em `models/emotion_cnn/weights/best_model.pth`.

### Download Automático dos Datasets

Os diretórios `dataset/` não são versionados no Git devido ao tamanho dos arquivos. Use o script `download_datasets.py` para baixar e preparar automaticamente todos os datasets via Kaggle API:

```bash
# Baixar todos os datasets (RWF-2000 + AffectNet)
python download_datasets.py --all

# Baixar apenas um dataset
python download_datasets.py --rwf2000
python download_datasets.py --affectnet
```

**Recursos do script:**
- Download com barra de progresso e retomada (pula arquivos já existentes)
- Extração robusta de ZIPs: corrige nomes em UTF-8/Cirílico (ex.: vídeos do RWF-2000), evita path traversal e trunca nomes muito longos
- Baixa o **balanced-affectnet** (`dollyprajapati182/balanced-affectnet`) já extraído na estrutura `{train,val,test}/<Classe>/*.png`

## Arquitetura dos Modelos

O projeto implementa múltiplas arquiteturas para diferentes aspectos da detecção de violência:

### 1. ResNet-LSTM (Modelo Base de Vídeo)

Arquitetura híbrida para análise temporal de vídeo:

```
┌─────────────┐
│   Frames    │ (16 frames por vídeo)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   ResNet-18     │ (pré-treinada no ImageNet)
│  Feature Ext.   │ → 512 dimensões por frame
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│      LSTM       │ (2 camadas, hidden_size=256)
│  Temporal Model │ → Modela dependências temporais
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   FC Layer      │ (classificação binária)
│   Output: 2     │ → [non-violent, violent]
└─────────────────┘
```

**Componentes:**
- ResNet-18 pré-treinada no ImageNet (extrai 512 dims por frame)
- LSTM com 2 camadas (hidden_size=256) para modelagem temporal
- Camada FC final para classificação binária

### 2. CNN 3D (Action Recognition)

Modelos 3D para reconhecimento de ações em vídeo:

- **R3D-18**: ResNet 3D com convoluções 3D puras
- **R(2+1)D-18**: ResNet com convoluções factorizadas (2D+1D)
- **MC3-18**: Mixed Convolution 3D

### 3. EmotionNet (Reconhecimento de Emoções)

Modelo baseado em **DeiT-Small** (Data-efficient Image Transformer) para classificação de emoções faciais:

```
┌───────────────────┐
│  Face extraída    │ (224×224, RGB)
└─────────┬─────────┘
          ▼
┌─────────────────────────┐
│   DeiT-Small Backbone   │ (Vision Transformer pré-treinado no ImageNet)
│   22M params, embed=384 │ → 384 dims por face
│   depth=12, heads=6     │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Classifier (2-layer)  │ Linear(384→128) → ReLU → Linear(128→8)
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Output: 8 emoções     │ (probabilidades)
└─────────────────────────┘
```

**Componentes:**
- Backbone: DeiT-Small (`deit_small_patch16_224`) pré-treinada no ImageNet, via `timm` (384 dims, 12 camadas, 6 heads, patch 16)
- Classifier: `Linear(384 → 128) → ReLU → Linear(128 → 8)` com Dropout(0.3)
- Treinado em: Balanced-AffectNet (via `train_emotion_model.py`)

**Técnicas de treinamento:**
- **Focal Loss** (γ configurável) para lidar com desbalanceamento de classes
- **WeightedRandomSampler** opcional via `--balance-classes` (OFF por padrão — o balanced-affectnet já é balanceado)
- **AdamW** com warmup linear + cosine annealing
- **Mixed precision (AMP)**, gradient clipping e early stopping
- Suporte a `--resume` para retomar treino do último checkpoint

**Classes de Emoção:**
- Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise, Contempt

### 4. MultimodalRiskDetector (Modelo Principal)

Arquitetura multimodal que combina todas as modalidades:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Video Features │  │  Pose Features  │  │ Emotion Features│
│  (ResNet-LSTM)  │  │  (MediaPipe)    │  │  (EmotionNet)   │
│  T × 256        │  │  T × 99         │  │  T × 128        │
│  (CNN3D: T×512) │  │                 │  │  (embeddings)   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Video Processor │  │ Pose Processor  │  │Emotion Processor│
│ (projeção linear)│ │ (LSTM)          │  │ (LSTM)          │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Fusion Module  │
                     │ (Cross-Attention│
                     │  3 heads)       │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Classifier     │
                     │  (Binary)       │
                     └─────────────────┘
```

**Estratégia de Fusão (única):**
- **Cross-Attention**: Vídeo (query) atende às memórias de pose + emoção via Multi-Head Attention (3 heads). A saída é agregada e classificada. Fusion `early`/`late` foram removidos — `fusion_method` agora aceita apenas `cross_attention`.

**Modalidades:**
- **Vídeo**: Features extraídas do CNN 3D R2Plus1D (512 dims, **padrão**) ou ResNet-LSTM (256 dims) — usadas como *token* de consulta
- **Pose**: 33 keypoints do MediaPipe (99 dims: x, y, visibility)
- **Emoção**: embeddings de 128 dims da penúltima camada do EmotionNet (não mais probabilidades de 8 classes)

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git (para clonar o repositório)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/murilo-rolo/vc-seguranca.git
   cd vc-seguranca
   ```

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt --prefer-binary
   ```
   > **Nota para Python 3.14:** Use `--prefer-binary` para garantir instalação de wheels pré-compilados e evitar erros de compilação.

4. **Baixe os datasets necessários:**

   O projeto inclui o script `download_datasets.py`, que baixa e prepara automaticamente todos os datasets:
   
   ```bash
   # RWF-2000 (obrigatório - dataset principal)
   python download_datasets.py --rwf2000
   
   # AffectNet (opcional - para treinar o EmotionNet)
   python download_datasets.py --affectnet
   
   # Ou baixe tudo de uma vez
   python download_datasets.py --all
   ```
   
   > **Alternativa manual:** baixe os datasets de uma fonte confiável (Kaggle, GitHub, etc.) e extraia em `dataset/` seguindo a estrutura mencionada acima.

## Uso

### 1. Pré-processamento

Todo o pré-processamento é feito por **um único script** (`run_preprocessing.py`) com subcomandos:

| Subcomando | Descrição |
|---|---|
| `organize` | Organiza os vídeos do RWF-2000 em `data/raw` |
| `frames` | Extrai N frames por vídeo, redimensiona e normaliza em `data/processed` |
| `pose` | Extrai keypoints de pose (MediaPipe) em `data/pose` |
| `emotion` | Extrai vetores de emoção (EmotionNet) em `data/emotion` |
| `all` | Executa todas as etapas em sequência |

#### Passo 1: Organizar vídeos

Organiza os vídeos do dataset RWF-2000:

```bash
python run_preprocessing.py organize
```

Ou manualmente:
```bash
python -m src.preprocessing.organize_videos
```

#### Passo 2: Extrair frames de vídeo

Extrai N frames por vídeo, redimensiona e normaliza:

```bash
python run_preprocessing.py frames --num_frames 16
# ou
python -m src.preprocessing.extract_frames
```

**Opções:**
- `--num_frames`: Número de frames por vídeo (padrão 16)
- `--target_size`: Tamanho (altura largura) dos frames (padrão `112 112`)
- `--normalize` / `--no-normalize`: Normaliza pixels para [0, 1] (padrão: ligado)
- `--workers`: Workers paralelos para extração

#### Passo 3: Extrair keypoints de pose

Extrai keypoints de pose usando MediaPipe:

```bash
python run_preprocessing.py pose --num_frames 16
```

**Opções:**
- `--num_frames`: Número de frames a processar por vídeo (None = todos)
- `--min_detection_confidence`: Confiança mínima de detecção (padrão 0.5; 0.5-0.7 recomendado)
- `--min_tracking_confidence`: Confiança mínima de rastreamento (padrão 0.5)
- `--model_complexity`: `0` (Lite), `1` (Full, padrão) ou `2` (Heavy)

#### Passo 4: Extrair emoções faciais

Extrai vetores de emoção usando EmotionNet (DeiT-Small). Cada vídeo gera uma sequência de embeddings de 128 dims (saída da penúltima camada, não probabilidades de classes), salvos como `emotions/emotions.npy` (formato `T × 128`):

```bash
# Primeiro, treine o modelo de emoção (se ainda não tiver)
python train_emotion_model.py --epochs 60

# Depois, extraia emoções do RWF-2000
python run_preprocessing.py emotion
```

O modelo é carregado automaticamente de `models/emotion_cnn/weights/best_model.pth`.
Se o checkpoint não existir, o script usa pesos ImageNet (modelo não treinado em emoções).
Se não houver rostos detectados em um vídeo, é usado o embedding neutro (128 dims) em cache em `models/emotion_cnn/weights/neutral_embedding.npy`.

**Opções adicionais:**
- `--face_detector`: `mtcnn` (padrão), `retinaface` ou `haar`
- `--face_aggregation`: `mean` (padrão) ou `max` — como agregar **todas** as faces detectadas em um frame em um único embedding `(128,)` (cenas de briga/multidão com múltiplas faces)
- `--aggregation`: `mean` (padrão) ou `max` (agregação temporal ao longo dos frames)
- `--num_frames`: Número de frames por vídeo (None = todos)
- `--device`: `cuda` (padrão, se disponível) ou `cpu`

#### Pipeline completo

Executa todas as etapas em sequência (organize → frames → pose → emotion):

```bash
python run_preprocessing.py all --num_frames 16
```

**Configuração customizada:**

```python
from src.preprocessing import preprocess_dataset

preprocess_dataset(
    raw_data_root="data/raw",
    processed_data_root="data/processed",
    num_frames=16,          # Número de frames por vídeo
    target_size=(112, 112), # Tamanho dos frames
    normalize=True          # Normalização dos pixels
)
```

### 2. Treinamento

O projeto oferece múltiplos scripts de treinamento para diferentes modelos:

#### 2.1. Treinamento ResNet-LSTM (Modelo Base)

Treinamento do modelo ResNet-LSTM para análise de vídeo:

```bash
python -m src.training.train \
    --batch_size 8 \
    --num_frames 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3
```

#### 2.2. Treinamento EmotionNet

Treina o modelo de reconhecimento de emoções (DeiT-Small) no Balanced-AffectNet:

```bash
python train_emotion_model.py \
    --epochs 60 \
    --batch_size 32 \
    --learning_rate 1e-5
```

**Opções principais:**
- `--epochs`: Número de épocas (padrão: 60)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--learning_rate`: Learning rate (padrão: 1e-5)
- `--dataset_path`: Caminho do dataset de emoção (padrão: `dataset/balanced-affectnet`; fallback: `dataset/AffectNet`)
- `--balance-classes`: Usa WeightedRandomSampler para balancear os batches (**OFF por padrão** — o dataset já é balanceado)
- `--min_confidence`: Confiança mínima do label no `labels.csv` (padrão: 0.7; **aplica-se apenas a datasets CSV legados**)
- `--focal-gamma`: Gamma da Focal Loss (padrão: 2)
- `--weight_decay`: Weight decay do AdamW (padrão: 0.05)
- `--warmup_epochs`: Épocas de warmup linear (padrão: 5)
- `--grad-clip`: Gradient clipping max norm (padrão: 1.0; `0` desativa)
- `--resume`: Retoma o treino a partir do último `best_model.pth`
- `--amp` / `--no-amp`: Ativa/desativa mixed precision

O dataset de emoção é localizado automaticamente em `dataset/balanced-affectnet` (splits `train`/`val`; se ausente, o AffectNet legado em `dataset/AffectNet` é usado como fallback). O modelo é salvo em `models/emotion_cnn/weights/best_model.pth`.

#### 2.3. Treinamento CNN 3D em RWF-2000 (2 classes)

O backbone é pré-treinado no **Kinetics400** (carregado automaticamente via torchvision) e o script faz o fine-tuning em RWF-2000:

```bash
python train_cnn3d.py \
    --epochs 30 \
    --batch_size 8
```

**Opções principais:**
- `--model_name`: `r3d_18`, `r2plus1d_18` (padrão) ou `mc3_18`
- `--freeze_backbone`: Congela o backbone durante o fine-tuning (treina apenas o classifier)
- `--clip_size`: Tamanho do clipe H W (padrão: `112 112`)

Os pesos são salvos em `models/cnn3d/weights/best_model.pth`.

#### 2.4. Treinamento Multimodal (Modelo Principal)

Treina o modelo multimodal completo combinando vídeo, pose e emoção:

```bash
python train_multimodal.py \
    --epochs 50 \
    --batch_size 8 \
    --use_temporal_modeling
```

**Parâmetros principais:**
- `--use_temporal_modeling`: Usa LSTM para modelagem temporal de pose e emoção
- `--video_backbone`: Backbone de vídeo do multimodal — `cnn3d` (padrão) ou `resnet_lstm`
- `--video_model_path`: Caminho para o backbone de vídeo pré-treinado (padrão: `models/cnn3d/weights/best_model.pth` se `--video_backbone cnn3d`, `models/resnet_lstm/weights/best_model.pth` se `resnet_lstm`)
- `--window_size`: Tamanho da janela temporal (padrão: 16)

A fusão é sempre **cross-attention** (sem flag `--fusion_method`). O checkpoint salva `fusion_method=cross_attention`, `emotion_feature_dim=128`, `video_backbone` e `video_feature_dim` (512 para CNN 3D, 256 para ResNet-LSTM), usados automaticamente na inferência e avaliação.

O modelo é salvo em `models/multimodal/weights/best_model.pth`.

#### Parâmetros Principais

| Parâmetro | Descrição | Padrão | Recomendação |
|-----------|-----------|--------|--------------|
| `--batch_size` | Tamanho do batch | 8 | 4-8 para GPU, 2-4 para CPU |
| `--num_frames` | Frames por vídeo | 16 | 8-32 dependendo do vídeo |
| `--num_epochs` | Número de épocas | 50 | 10 para teste, 50+ para produção |
| `--learning_rate` | Taxa de aprendizado | 1e-4 | 1e-4 a 1e-3 |
| `--hidden_size` | Tamanho do hidden state LSTM | 256 | 128-512 |
| `--num_layers` | Camadas LSTM | 2 | 1-3 |
| `--dropout` | Taxa de dropout | 0.3 | 0.3-0.5 |
| `--num_workers` | Workers do DataLoader | 4 | 0-4 dependendo da CPU |
| `--early_stopping_patience` | Paciência do early stopping | 10 | 5-15 |

**Dicas para computadores com poucos recursos:**
- Use `--batch_size 4` ou `--batch_size 2` para reduzir uso de memória
- Use `--num_workers 0` ou `--num_workers 2` para reduzir uso de CPU
- Use `--num_epochs 10` para testes rápidos
- Use `--device cpu` se tiver problemas com GPU

O melhor modelo será salvo automaticamente em `models/resnet_lstm/weights/best_model.pth`.

### 3. Avaliação

O script `run_evaluation.py` executa um pipeline completo de experimentos de avaliação. O tipo de modelo é selecionado com `--model`:

#### 3.1. Avaliação de Modelo Unimodal (ResNet-LSTM)

```bash
python run_evaluation.py \
    --model baseline \
    --model_path models/resnet_lstm/weights/best_model.pth
```

#### 3.2. Avaliação de Modelo Multimodal

```bash
python run_evaluation.py \
    --model multimodal \
    --model_path models/multimodal/weights/best_model.pth
```

#### 3.3. Avaliação de Modelo CNN 3D

```bash
python run_evaluation.py \
    --model cnn3d \
--model_path models/cnn3d/weights/best_model.pth
```

Os resultados do CNN 3D são salvos em `results/cnn3d/`.

#### 3.4. Avaliação por Sub-Modelo (F5)

Avalia cada branch do modelo multimodal **independentemente** — `video` (CNN 3D por padrão; use `--video_backbone resnet_lstm` para ResNet-LSTM), `pose` (branch LSTM de pose do multimodal) e `emotion` (EmotionNet, métricas sobre as **8 classes**):

```bash
python run_evaluation.py --model video   --model_path models/cnn3d/weights/best_model.pth
python run_evaluation.py --model video   --video_backbone resnet_lstm --model_path models/resnet_lstm/weights/best_model.pth
python run_evaluation.py --model pose    --model_path models/multimodal/weights/best_model.pth
python run_evaluation.py --model emotion --model_path models/emotion_cnn/weights/best_model.pth

# Avaliar os três de uma vez e agregar em um único relatório
python run_evaluation.py --model multimodal --model_path models/multimodal/weights/best_model.pth --all
```

As métricas de cada sub-modelo são salvas em `results/{video,pose,emotion}/metrics.json` e o relatório agregado em `results/reports/per_model_evaluation.json`.

#### 3.5. Experimentos de Avaliação

Além das métricas básicas, o script pode executar outros experimentos:

```bash
# Executar todos os experimentos
python run_evaluation.py \
    --model multimodal \
    --model_path models/multimodal/weights/best_model.pth \
    --all

# Executar experimentos específicos
python run_evaluation.py --model baseline --model_path <modelo> --metrics
python run_evaluation.py --model baseline --model_path <modelo> --robustness
python run_evaluation.py --model baseline --model_path <modelo> --performance
python run_evaluation.py --model baseline --model_path <modelo> --limitations

# Gerar gráficos padronizados em results/charts/
python run_evaluation.py --model multimodal --model_path <modelo> --all --charts

# Estudo de impacto das modalidades no modelo fusionado (0 = sanity check)
python run_evaluation.py --model multimodal --model_path <modelo> --impact_study --noise_std 0.05
```

**Parâmetros:**
- `--model`: `baseline`, `cnn3d`, `multimodal`, `video`, `pose` ou `emotion` (obrigatório)
- `--model_path`: Caminho para checkpoint do modelo (obrigatório; para `video`/`pose`/`emotion` use os flags específicos abaixo)
- `--video_backbone`: Backbone do sub-modelo `video` — `cnn3d` (padrão) ou `resnet_lstm` (também usado para extrair features do multimodal)
- `--video_model_path` / `--pose_model_path` / `--emotion_model_path`: checkpoints explícitos dos sub-modelos (padrão: `models/<modelo>/weights/best_model.pth`)
- `--metrics`: Métricas básicas (padrão se nenhum experimento for especificado)
- `--robustness`: Testa robustez a distorções (blur, ruído, baixa iluminação, oclusão, variação de resolução)
- `--performance`: Mede FPS, latência e uso de recursos
- `--limitations`: Analisa falsos positivos, falsos negativos e casos limítrofes
- `--charts`: Gera gráficos padronizados (barras accuracy/F1, heatmaps de confusão, curvas ROC/PR) em `results/charts/`
- `--impact_study`: Estudo de impacto das modalidades no modelo fusionado (perturba cada input com ruído gaussiano e mede o delta do logit e a queda de accuracy)
- `--noise_std`: Desvio padrão do ruído no estudo de impacto (padrão: 0.05; `0` = sanity check)
- `--experiment_name`: Nome do experimento (padrão: tipo do modelo)

Os resultados são salvos em `results/experiments/<experiment_name>/`, incluindo:
- `metrics/` — relatórios de métricas, curvas ROC e PR
- `robustness/` — resultados por distorção e intensidade
- `performance/` — benchmarks de FPS/latência
- `limitations/` — exemplos de erros com imagens
- `impact_study/` — `impact_study.json` e `impact_ranking.json` (quando `--impact_study`)
- `evaluation_summary.json` — resumo geral

### 3.6. Comparação Baseline vs Multimodal

O script `compare_baseline_multimodal.py` avalia ambos os modelos no mesmo conjunto de teste e compara as métricas:

```bash
python compare_baseline_multimodal.py \
    --baseline_model_path models/resnet_lstm/weights/best_model.pth \
    --multimodal_model_path models/multimodal/weights/best_model.pth
```

O resultado da comparação (incluindo a melhoria de accuracy e F1) é salvo em `results/comparison/comparison.json`.

### 4. Métricas de Avaliação

O script de avaliação calcula as seguintes métricas:

- **Accuracy**: Acurácia geral do modelo
- **Precision**: Precisão por classe (violent/non-violent)
- **Recall**: Recall por classe (sensibilidade)
- **F1-Score**: F1-score por classe (média harmônica)
- **Matriz de Confusão**: Visualização de erros de classificação
- **ROC-AUC**: Área sob a curva ROC
- **PR-AUC**: Área sob a curva Precision-Recall

### 5. Inferência em Tempo Real

Execute detecção de violência em tempo real usando webcam ou stream RTSP:

```bash
python run_realtime_risk_detection.py \
    --multimodal_model models/multimodal/weights/best_model.pth \
    --emotion_model models/emotion_cnn/weights/best_model.pth \
    --source 0 \
    --risk_threshold 0.8 \
    --consecutive_windows 3
```

O backbone de vídeo padrão é o **CNN 3D** (checkpoint em `models/cnn3d/weights/best_model.pth`). Para usar ResNet-LSTM, informe `--video_backbone resnet_lstm --video_model models/resnet_lstm/weights/best_model.pth`.

**Parâmetros:**
- `--source`: `0` para webcam ou URL RTSP (ex: `rtsp://...`)
- `--risk_threshold`: Threshold de probabilidade para alerta (0.0-1.0)
- `--consecutive_windows`: Número de janelas consecutivas acima do threshold para alerta
- `--window_size`: Tamanho da janela temporal (padrão: 16)
- `--overlap`: Sobreposição entre janelas (padrão: 8)
- `--frame_size`: Tamanho dos frames para processamento H W (padrão: `224 224`)
- `--face_aggregation`: `mean` (padrão) ou `max` — agregação de **todas** as faces de um frame em um único embedding de emoção
- `--video_backbone`: `cnn3d` (padrão) ou `resnet_lstm` — backbone de vídeo do multimodal
- `--cnn3d_model`: Caminho para modelo CNN 3D (padrão: `models/cnn3d/weights/best_model.pth`)
- `--video_model`: Caminho para modelo ResNet-LSTM (usado se `--video_backbone resnet_lstm`)
- `--no_display`: Não exibir o vídeo (apenas processar)
- `--device`: Device para inferência (`cuda`/`cpu`)

O `video_backbone` e o `video_feature_dim` do modelo multimodal são lidos do próprio checkpoint — não é preciso informá-los na linha de comando.

## Estrutura de Dados

O projeto utiliza uma estrutura padronizada para organizar dados brutos e processados. É importante seguir esta estrutura para garantir que todos os scripts funcionem corretamente.

### Estrutura Completa de Diretórios

```
vc-seguranca/
├── dataset/                           # Datasets originais (não versionados)
│   ├── RWF-2000/                      # Dataset principal (violência CCTV)
│   │   ├── train/
│   │   │   ├── Fight/                 # Vídeos violentos (treino)
│   │   │   │   └── *.avi
│   │   │   └── NonFight/              # Vídeos não violentos (treino)
│   │   │       └── *.avi
│   │   └── val/
│   │       ├── Fight/                  # Vídeos violentos (validação)
│   │       └── NonFight/               # Vídeos não violentos (validação)
│   └── balanced-affectnet/            # Dataset de emoções (8 classes)
│       ├── train/
│       │   ├── Anger/
│       │   ├── Happy/
│       │   └── ... (8 classes)
│       ├── val/
│       └── test/
│
├── data/                              # Dados processados
│   ├── raw/                           # Vídeos organizados (após organize_videos.py)
│   │   ├── violent/                   # Todos os vídeos violentos (Fight)
│   │   │   └── *.avi
│   │   └── non_violent/               # Todos os vídeos não violentos (NonFight)
│   │       └── *.avi
│   │
│   ├── processed/                     # Frames extraídos (após extract_frames.py)
│   │   ├── violent/                   # Frames de vídeos violentos
│   │   │   └── <video_id>/            # Pasta por vídeo
│   │   │       ├── frame_0000.jpg
│   │   │       ├── frame_0001.jpg
│   │   │       └── ... (16 frames)
│   │   └── non_violent/               # Frames de vídeos não violentos
│   │       └── <video_id>/
│   │           └── ...
│   │
│   ├── pose/                          # Keypoints de pose (após extract_pose.py)
│   │   ├── rwf2000/                   # Pose do RWF-2000
│   │   │   ├── train/
│   │   │   │   ├── violent/           # Pose de vídeos violentos (treino)
│   │   │   │   │   └── <video_id>.npy # Shape: (num_frames, 33, 3)
│   │   │   │   └── non_violent/       # Pose de vídeos não violentos (treino)
│   │   │   │       └── <video_id>.npy
│   │   │   └── val/                   # Pose de vídeos (validação)
│   │   │       ├── violent/
│   │   │       └── non_violent/
│   │
│   └── emotion/                       # Vetores de emoção (após extract_emotion.py)
│       └── rwf2000/                   # Emoções do RWF-2000
│           ├── train/
│           │   ├── violent/           # Emoções de vídeos violentos (treino)
│           │   │   └── <video_id>.npy # Shape: (num_frames, 128)
│           │   └── non_violent/       # Emoções de vídeos não violentos (treino)
│           │       └── <video_id>.npy
│           └── val/                   # Emoções de vídeos (validação)
│               ├── violent/
│               └── non_violent/
```

### Convenções de Nomenclatura

**Vídeos:**
- **Dataset original**: `dataset/RWF-2000/{split}/{Fight|NonFight}/<video_name>.avi`
- **Após organização**: `data/raw/{violent|non_violent}/<video_name>.avi`
- **ID do vídeo**: Nome do arquivo sem extensão (ex: `video_0001`)

**Frames:**
- **Estrutura**: `data/processed/{violent|non_violent}/<video_id>/frame_XXXX.jpg`
- **Formato**: `frame_0000.jpg`, `frame_0001.jpg`, ..., `frame_0015.jpg` (16 frames)

**Pose (Keypoints):**
- **Estrutura**: `data/pose/rwf2000/{split}/{violent|non_violent}/<video_id>.npy`
- **Formato**: Array NumPy com shape `(num_frames, 33, 3)`
  - 33 keypoints do MediaPipe
  - 3 valores: (x, y, visibility)
- **Exemplo**: `data/pose/rwf2000/train/violent/video_0001.npy`

**Emoção:**
- **Estrutura**: `data/emotion/rwf2000/{split}/{violent|non_violent}/<video_id>.npy`
- **Formato**: Array NumPy com shape `(num_frames, 128)`
  - Embeddings de 128 dims da penúltima camada do EmotionNet (não probabilidades de classes)
  - Por frame: agregação de **todas** as faces detectadas (mean/max, `--face_aggregation`); sem faces → embedding neutro
- **Exemplo**: `data/emotion/rwf2000/train/violent/video_0001.npy`

## Estrutura do Projeto

```
vc-seguranca/
├── dataset/                    # Datasets (não versionados)
│   ├── RWF-2000/              # Dataset principal (violência CCTV)
│   └── balanced-affectnet/    # Dataset de emoções (8 classes)
├── data/                      # Dados processados
│   ├── raw/                   # Vídeos organizados
│   ├── processed/             # Frames extraídos
│   ├── pose/                  # Keypoints de pose
│   └── emotion/               # Vetores de emoção
├── models/                    # Modelos treinados (pesos + experimentos)
│   ├── resnet_lstm/           # ResNet-18 + LSTM
│   │   ├── weights/           # best_model.pth
│   │   └── experiments/       # Logs e métricas de treinamento
│   ├── emotion_cnn/           # EmotionNet (DeiT-Small)
│   │   ├── weights/           # best_model.pth, confusion_matrix.npy
│   │   └── experiments/       # training_history.json, confusion_matrix.png
│   ├── cnn3d/                 # CNN 3D (R3D, R(2+1)D, MC3)
│   │   ├── weights/ # Pesos do finetuning
│   │   └── experiments/
│   └── multimodal/            # MultimodalRiskDetector
│       ├── weights/
│       └── experiments/
├── src/
│   ├── paths.py                # Gerenciamento centralizado de caminhos
│   ├── preprocessing/          # Pré-processamento
│   │   ├── organize_videos.py
│   │   └── extract_frames.py
│   ├── pose/                  # Extração de pose
│   │   ├── extract_pose.py
│   │   └── pose_dataset.py
│   ├── emotion/               # Extração de emoção
│   │   ├── extract_emotion.py
│   │   └── emotion_dataset.py
│   ├── datasets/              # Datasets e DataLoaders
│   │   ├── surveillance_dataset.py
│   │   ├── multimodal_dataset.py
│   │   └── video3d_dataset.py
│   ├── models/                # Modelos de Deep Learning
│   │   ├── resnet_lstm.py     # ResNet-LSTM
│   │   ├── cnn3d_risk.py      # CNN 3D
│   │   ├── emotion_cnn.py     # EmotionNet (DeiT-Small)
│   │   ├── multimodal_risk.py # MultimodalRiskDetector
│   │   └── losses.py          # Focal Loss
│   ├── training/              # Scripts de treinamento
│   │   ├── train.py           # Treinamento ResNet-LSTM
│   │   └── utils.py           # Funções compartilhadas (run_epoch, dataloader, etc.)
│   ├── evaluation/            # Avaliação experimental
│   │   ├── metrics.py         # Métricas básicas (acc, precision, AUC-ROC, PR)
│   │   ├── robustness_eval.py # Robustez a distorções
│   │   ├── performance_eval.py# Performance (FPS, latência)
│   │   ├── limitations_analysis.py # FPs, FNs, casos limítrofes
│   │   ├── ablation_study.py  # Estudo de ablação
│   │   └── utils.py           # Aplicação de distorções e utilitários
│   └── inference/             # Inferência
│       ├── realtime_risk_detector.py
│       └── multi_camera_detector.py
├── examples/                  # Scripts de exemplo
│   ├── example_usage.py
│   ├── example_cnn3d_usage.py
│   └── ... (outros exemplos)
├── results/                   # Resultados de experimentos
│   ├── experiments/           # Resultados do run_evaluation.py
│   ├── comparison/            # Comparação baseline vs multimodal
│   └── reports/               # Relatórios
├── download_datasets.py       # Download dos datasets (Kaggle API)
├── compare_baseline_multimodal.py  # Comparação baseline vs multimodal
├── train_*.py                 # Scripts de treinamento (raiz)
├── run_*.py                   # Scripts de execução (raiz)
├── requirements.txt           # Dependências
└── README.md                  # Este arquivo
```

## Requisitos de Sistema

### Mínimos

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **RAM**: 8GB
- **Espaço em disco**: ~10GB (dataset + dados processados)
- **CPU**: Qualquer processador moderno

### Recomendados

- **GPU**: NVIDIA com CUDA compatível (para treinamento mais rápido)
- **RAM**: 16GB+
- **Espaço em disco**: 20GB+ (para múltiplos experimentos)
- **CPU**: Multi-core para pré-processamento paralelo

### Dependências Principais

- `torch>=2.0.0` - Framework de Deep Learning
- `torchvision>=0.15.0` - Modelos pré-treinados e transformações
- `timm>=1.0.0` - Modelos Vision Transformer (backbone do EmotionNet)
- `opencv-python>=4.8.0` - Processamento de vídeo
- `numpy>=2.4.0` - Operações numéricas (Python 3.14 requer >=2.4.0)
- `scikit-learn>=1.3.0` - Métricas de avaliação
- `tqdm>=4.65.0` - Barras de progresso
- `Pillow>=10.0.0` - Processamento de imagens
- `mediapipe>=0.10.0` - Detecção de pose
- `facenet-pytorch>=2.5.0` - Detecção de faces
- `matplotlib>=3.7.0` - Plotagem de gráficos (curvas ROC, PR)
- `seaborn>=0.12.0` - Visualização de matriz de confusão

## Troubleshooting

### Problemas Comuns

**1. Erro de memória durante o treinamento**
- Solução: Reduza o `batch_size` para 2 ou 4
- Solução: Reduza o `num_frames` para 8

**2. Erro ao instalar numpy no Python 3.14 (compilação do código-fonte)**
- Problema: NumPy 2.2.x e anteriores tentam compilar do código-fonte no Python 3.14
- Solução: Use `pip install -r requirements.txt --prefer-binary` para forçar wheels pré-compilados
- Solução alternativa: O `requirements.txt` já especifica `numpy>=2.4.0`, que tem wheels para Python 3.14

**3. Dataset não encontrado**
- Verifique se o dataset está em `dataset/RWF-2000/`
- Verifique a estrutura de pastas (train/Fight, train/NonFight, etc.)

**4. Erro ao baixar pesos do ImageNet**
- Verifique sua conexão com a internet
- O PyTorch baixará automaticamente na primeira execução

**5. Treinamento muito lento**
- Use GPU se disponível: `--device cuda`
- Reduza `num_workers` se estiver usando CPU
- Considere reduzir `num_frames` ou `batch_size`

**6. Erro de importação de módulos**
- Certifique-se de que o ambiente virtual está ativado
- Reinstale as dependências: `pip install -r requirements.txt`

**7. Erro ao criar o EmotionNet (módulo `timm` ausente)**
- O EmotionNet usa DeiT-Small via `timm`; instale com `pip install timm>=1.0.0`
- Na primeira execução o backbone DeiT-Small baixa os pesos do ImageNet automaticamente

## Pipeline Completo de Execução

### Opção 1: Pipeline Automatizado (Recomendado)

Use o script master `train_pipeline.py` para automatizar todo o processo:

```bash
# Treinar tudo do zero (pipeline completo)
python train_pipeline.py --all

# Treinar apenas modelos base
python train_pipeline.py --base_models

# Treinar apenas multimodal (assumindo modelos base já existem)
python train_pipeline.py --multimodal

# Treinar apenas um modelo específico
python train_pipeline.py --resnet_lstm
python train_pipeline.py --emotion
python train_pipeline.py --cnn3d

# Treinar com opções customizadas
python train_pipeline.py --all --skip_emotion --skip_cnn3d --epochs 30 --batch_size 4

# Forçar retreinamento mesmo se modelos já existirem
python train_pipeline.py --all --force_retrain

# Caminhos customizados de datasets
python train_pipeline.py --all --affectnet_path /caminho/balanced-affectnet
```

**Vantagens do script master:**
- ✅ Valida pré-requisitos automaticamente
- ✅ Detecta modelos já treinados e pergunta se deseja retreinar
- ✅ Orquestra toda a sequência de treinamento
- ✅ Fornece relatório detalhado ao final
- ✅ Trata erros e permite continuar com etapas opcionais

### Opção 2: Pipeline Manual (Passo a Passo)

Se preferir executar manualmente:

1. **Pré-processamento de Dados**
   ```bash
   # 1. Organizar vídeos e extrair frames
   python run_preprocessing.py organize
   python run_preprocessing.py frames --num_frames 16
   
   # 2. Extrair pose
   python run_preprocessing.py pose --num_frames 16
   
   # 3. Treinar EmotionNet e extrair emoções
   python train_emotion_model.py
   python run_preprocessing.py emotion
   ```

2. **Treinamento de Modelos Base**
   ```bash
   # 1. Treinar ResNet-LSTM (usado como baseline)
   python -m src.training.train --epochs 50
   
   # 2. Treinar CNN 3D (backbone de vídeo padrão do multimodal)
   python train_cnn3d.py
   
   # 3. Treinar EmotionNet (se ainda não tiver)
   python train_emotion_model.py
   ```

3. **Treinamento Multimodal**
   ```bash
   # Backbone de vídeo padrão: CNN 3D (models/cnn3d/weights/best_model.pth)
   python train_multimodal.py \
       --epochs 50
   
   # Alternativa: usar ResNet-LSTM como backbone de vídeo
   python train_multimodal.py \
       --epochs 50 \
       --video_backbone resnet_lstm \
       --video_model_path models/resnet_lstm/weights/best_model.pth
   ```

4. **Avaliação**
   ```bash
   python run_evaluation.py \
       --model multimodal \
       --model_path models/multimodal/weights/best_model.pth
   ```

5. **Inferência em Tempo Real**
   ```bash
   # Backbone de vídeo padrão: CNN 3D
   python run_realtime_risk_detection.py \
       --multimodal_model models/multimodal/weights/best_model.pth \
       --emotion_model models/emotion_cnn/weights/best_model.pth
   ```

### Ordem de Execução Recomendada

**Sequência completa:**
1. Pré-processamento → 2. Modelos Base → 3. Multimodal → 4. Avaliação → 5. Inferência

**Dependências:**
- Multimodal requer: CNN 3D (obrigatório — backbone de vídeo padrão) ou ResNet-LSTM (alternativo via `--video_backbone resnet_lstm`), EmotionNet (opcional), Pose e Emotion extraídos
- EmotionNet requer: Balanced-AffectNet para treinamento (opcional)

## Referências

### Datasets
- **RWF-2000**: Real-World Fighting Dataset para detecção de violência
- **AffectNet (balanced)**: Facial Expression Dataset para reconhecimento de emoções

### Modelos e Arquiteturas
- **ResNet**: Deep Residual Learning for Image Recognition (He et al., 2015)
- **LSTM**: Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)
- **R(2+1)D**: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)
- **DeiT**: Training Data-efficient Image Transformers & Distillation through Attention (Touvron et al., 2021)
- **Focal Loss**: Focal Loss for Dense Object Detection (Lin et al., 2017)
- **MediaPipe**: Framework de ML para detecção de pose

### Ferramentas
- **PyTorch**: Framework de Deep Learning
- **OpenCV**: Biblioteca de Visão Computacional
- **MediaPipe**: Detecção de pose e landmarks
