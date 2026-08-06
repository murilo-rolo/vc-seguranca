# Multimodal Model

Modelo Multimodal para detecção de risco/violência combinando múltiplas modalidades.

## Estrutura

```
multimodal/
├── weights/          # Pesos treinados do modelo
│   └── best_model.pth
├── configs/          # Configurações de treinamento
│   └── config.json
└── experiments/      # Logs e métricas de treinamento
    └── training_history.json
```

## Modalidades

- **Video Features**: ResNet-LSTM (extração de features de vídeo)
- **Pose Features**: Keypoints de pose (MediaPipe)
- **Emotion Features**: Vetores de emoção facial

## Dataset Utilizado

- **RWF-2000**: Dataset principal
  - Localização: `dataset/RWF-2000/`
- **Dados processados**:
  - Vídeos: `data/processed/`
  - Pose: `data/pose/rwf2000/`
  - Emoção: `data/emotion/rwf2000/`

## Treinamento

```bash
python train_multimodal.py --epochs 50 --fusion_method late --batch_size 8
```

## Métodos de Fusão

- **Early Fusion**: Concatena features brutas antes do processamento
- **Late Fusion**: Processa cada modalidade separadamente e funde no final
- **Attention Fusion**: Fusão com mecanismo de atenção

## Pré-requisitos

1. Modelo ResNet-LSTM treinado (`models/resnet_lstm/weights/best_model.pth`)
2. Dados de pose e emoção processados
