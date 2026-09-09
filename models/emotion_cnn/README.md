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

- **Balanced-AffectNet**: Dataset de reconhecimento de emoções faciais (classificação binária)
  - Localização: `dataset/balanced-affectnet/` (`train`/`val`/`test` por split; fallback: `dataset/AffectNet/` legado com `labels.csv`)
  - Classes: Violent, Non-Violent

## Treinamento

```bash
# Dataset baixado por download_datasets.py --affectnet é detectado automaticamente
python train_emotion_model.py --epochs 60 --batch_size 32 --learning_rate 1e-5

# Caminho explícito / sampler balanceado (OFF por padrão — dataset já balanceado)
python train_emotion_model.py --dataset_path dataset/balanced-affectnet --balance-classes
```

## Arquitetura

- **Backbone**: DeiT-Small pré-treinada no ImageNet
- **Classifier**: Linear(384→128→8)
- **Loss Function**: Focal Loss (γ=2)
- **Otimizador**: AdamW com warmup + cosine annealing
