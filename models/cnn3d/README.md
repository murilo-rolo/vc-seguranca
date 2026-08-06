# CNN 3D Model

Modelo CNN 3D para detecção de violência em vídeos com duas etapas de treinamento.

## Estrutura

```
cnn3d/
├── weights/
│   ├── ucf101/       # Pesos do pré-treinamento
│   │   └── best_model.pth
│   └── rwf2000/      # Pesos do fine-tuning
│       └── best_model.pth
├── configs/          # Configurações de treinamento
│   └── config.json
└── experiments/
    ├── ucf101/       # Métricas do pré-treinamento
    │   └── training_history.json
    └── rwf2000/      # Métricas do fine-tuning
        └── training_history.json
```

## Datasets Utilizados

- **UCF101**: Pré-treinamento (9 classes relevantes)
  - Localização: `dataset/UCF101/`
- **RWF-2000**: Fine-tuning (classificação binária)
  - Localização: `dataset/RWF-2000/`

## Treinamento

```bash
# Etapa 1: Pré-treinamento em UCF101
python train_cnn3d.py --stage pretrain --model_name r2plus1d_18 --pretrained --epochs 50

# Etapa 2: Fine-tuning em RWF-2000
python train_cnn3d.py --stage finetune --pretrained_path models/cnn3d/weights/ucf101/best_model.pth --epochs 50

# Ou executar ambas as etapas
python train_cnn3d.py --stage both --pretrained --epochs 50
```

## Arquitetura

- **Backbones disponíveis**: r3d_18, r2plus1d_18, mc3_18
- **Pré-treinamento**: Kinetics400 (ImageNet weights)
- **Fine-tuning**: Transfer learning com backbone congelado ou não
