# Models Directory

Diretório contendo todos os modelos treinados do projeto, organizados por tipo.

## Estrutura

```
models/
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

Cada modelo tem seu script de treinamento na raiz:

```bash
# EmotionNet
python train_emotion_model.py

# CNN 3D
python train_cnn3d.py

# Multimodal
python train_multimodal.py
```

## Paths no Código

Os paths são definidos em `src/paths.py`:

```python
# Emotion CNN
EMOTION_CNN_WEIGHTS = MODELS_BASE / "emotion_cnn" / "weights"

# CNN 3D
CNN3D_WEIGHTS = MODELS_BASE / "cnn3d" / "weights"
CNN3D_EXPERIMENTS = MODELS_BASE / "cnn3d" / "experiments"

# Multimodal
MULTIMODAL_WEIGHTS = MODELS_BASE / "multimodal" / "weights"
```
