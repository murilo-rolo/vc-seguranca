# Emotion CNN Model (DeiT)

Modelo de Emotion Recognition baseado em DeiT-Small (Data-efficient Image Transformer).

## Estrutura

```
emotion_cnn/
├── weights/          # Pesos treinados do modelo
│   ├── best_model.pth
│   └── confusion_matrix.npy
├── configs/          # Configurações de treinamento
│   └── config.json
└── experiments/      # Logs e métricas de treinamento
    ├── training_history.json
    └── confusion_matrix.png
```

## Dataset Utilizado

- **AffectNet**: Dataset de reconhecimento de emoções faciais
  - Localização: `dataset/AffectNet/`
  - Classes: neutral, happy, sad, anger, fear, disgust, surprise, contempt

## Treinamento

```bash
python train_emotion_model.py --epochs 60 --batch_size 32 --learning_rate 1e-5
```

## Arquitetura

- **Backbone**: DeiT-Small pré-treinada no ImageNet
- **Classifier**: Linear(384→128→8)
- **Loss Function**: Focal Loss (γ=2)
- **Otimizador**: AdamW com warmup + cosine annealing
