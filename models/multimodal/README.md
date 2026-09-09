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

- **Video Features**: CNN 3D R2Plus1D (512 dims, **padrão**) ou ResNet-LSTM (256 dims) — token de consulta
- **Pose Features**: Keypoints de pose (YOLO26, 51 dims por frame: 17 joints × 3)
- **Emotion Features**: Embeddings de 128 dims (penúltima camada do EmotionNet) — não mais probabilidades de 8 classes. Por frame, **todas** as faces detectadas são agregadas em um único embedding (mean/max, `--face_aggregation`); sem faces → embedding neutro

## Dataset Utilizado

- **RWF-2000**: Dataset principal
  - Localização: `dataset/RWF-2000/`
- **Dados processados**:
  - Vídeos: `data/processed/`
  - Pose: `data/pose/rwf2000/`
  - Emoção: `data/emotion/rwf2000/`

## Treinamento

```bash
# Backbone de vídeo padrão: CNN 3D (models/cnn3d/weights/best_model.pth)
python train_multimodal.py --epochs 50 --batch_size 8

# Alternativa: ResNet-LSTM como backbone de vídeo
python train_multimodal.py --epochs 50 --batch_size 8 --video_backbone resnet_lstm --video_model_path models/resnet_lstm/weights/best_model.pth
```

## Método de Fusão

- **Cross-Attention** (único): Vídeo (query) atende às memórias de pose + emoção via Multi-Head Attention (3 heads). Fusion `early`/`late` foram removidos — `fusion_method` aceita apenas `cross_attention`.

O checkpoint salva `fusion_method`, `emotion_feature_dim` (128), `video_backbone` e `video_feature_dim` (512/256); a inferência e a avaliação leem esses valores automaticamente.

## Pré-requisitos

1. Backbone de vídeo treinado — CNN 3D por padrão (`models/cnn3d/weights/best_model.pth`; use `--video_backbone resnet_lstm` para ResNet-LSTM, `models/resnet_lstm/weights/best_model.pth`)
2. Dados de pose e emoção processados (emoção em formato `T × 128`)
