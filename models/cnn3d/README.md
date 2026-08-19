# CNN 3D Model

Modelo CNN 3D para detecção de violência em vídeos, pré-treinado no Kinetics400 e fine-tuned em RWF-2000.

## Estrutura

```
cnn3d/
├── weights/           # Pesos do fine-tuning
│   └── best_model.pth
├── configs/           # Configurações de treinamento
│   └── config.json
└── experiments/       # Métricas do fine-tuning
    └── training_history.json
```

## Dataset Utilizado

- **RWF-2000**: Fine-tuning (classificação binária violent/non-violent)
  - Localização: `dataset/RWF-2000/`

## Treinamento

O backbone (r3d_18, r2plus1d_18, mc3_18) é criado com `pretrained=True`, carregando automaticamente os pesos pré-treinados no **Kinetics400** via torchvision. O script faz apenas o fine-tuning em RWF-2000:

```bash
python train_cnn3d.py --model_name r2plus1d_18 --epochs 50
```

**Opções principais:**
- `--model_name`: `r3d_18`, `r2plus1d_18` (padrão) ou `mc3_18`
- `--freeze_backbone`: Congela o backbone durante o fine-tuning (treina apenas o classifier)
- `--epochs`, `--batch_size`, `--learning_rate`, `--num_frames`, `--clip_size`, `--dropout`

O melhor modelo é salvo em `models/cnn3d/weights/best_model.pth`.

## Arquitetura

- **Backbones disponíveis**: r3d_18, r2plus1d_18, mc3_18
- **Pré-treinamento**: Kinetics400 (via torchvision `weights_enum.DEFAULT`)
- **Fine-tuning**: Transfer learning com backbone congelado ou não

> A CNN 3D é o **backbone de vídeo padrão** do modelo multimodal (`--video_backbone cnn3d`); o checkpoint de fine-tuning em `models/cnn3d/weights/best_model.pth` é usado automaticamente no treinamento multimodal, na avaliação (`--model video`, `--model multimodal`) e na inferência em tempo real.

## Avaliação

```bash
python run_evaluation.py \
    --model cnn3d \
    --model_path models/cnn3d/weights/best_model.pth
```

As métricas, curvas ROC/PR e o resumo são salvos em `results/cnn3d/` (`metrics/`, `evaluation_summary.json`). O `--all` também inclui robustez a distorções, performance (FPS/latência) e análise de limitações.