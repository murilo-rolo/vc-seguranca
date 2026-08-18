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

> A CNN 3D é o **backbone de vídeo padrão** do modelo multimodal (`--video_backbone cnn3d`); o checkpoint de fine-tuning em `models/cnn3d/weights/rwf2000/best_model.pth` é usado automaticamente no treinamento multimodal, na avaliação (`--model video`, `--model multimodal`) e na inferência em tempo real.

## Avaliação

```bash
python run_evaluation.py \
    --model cnn3d \
    --model_path models/cnn3d/weights/rwf2000/best_model.pth
```

As métricas, curvas ROC/PR e o resumo são salvos em `results/cnn3d/` (`metrics/`, `evaluation_summary.json`). O `--all` também inclui robustez a distorções, performance (FPS/latência) e análise de limitações.
