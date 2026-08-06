# Models Directory

Diretório contendo todos os modelos treinados do projeto, organizados por tipo.

## Estrutura

```
models/
├── resnet_lstm/      # Modelo ResNet-18 + LSTM
├── emotion_cnn/      # Modelo DeiT para emoções
├── cnn3d/            # Modelo CNN 3D (R3D, R(2+1)D, MC3)
└── multimodal/       # Modelo Multimodal
```

## Organização por Modelo

Cada pasta de modelo contém:

- **weights/**: Pesos treinados (.pth, .pt)
- **experiments/**: Logs, métricas e resultados

## Compatibilidade

Esta estrutura é compatível com:
- Ambiente local
- Google Colab (via paths configuráveis em `src/paths.py`)

## Treinamento

Use o script principal para treinar modelos:

```bash
# Treinar todos os modelos
python train_pipeline.py --all

# Treinar modelos específicos
python train_pipeline.py --resnet_lstm
python train_pipeline.py --emotion
python train_pipeline.py --cnn3d
python train_pipeline.py --multimodal
```

## Paths no Código

Os paths são definidos em `src/paths.py`:

```python
# ResNet-LSTM
RESNET_LSTM_WEIGHTS = MODELS_BASE / "resnet_lstm" / "weights"

# Emotion CNN
EMOTION_CNN_WEIGHTS = MODELS_BASE / "emotion_cnn" / "weights"

# CNN 3D
CNN3D_UCF101_WEIGHTS = MODELS_BASE / "cnn3d" / "weights" / "ucf101"
CNN3D_RWF2000_WEIGHTS = MODELS_BASE / "cnn3d" / "weights" / "rwf2000"

# Multimodal
MULTIMODAL_WEIGHTS = MODELS_BASE / "multimodal" / "weights"
```
