# ResNet-LSTM Model

Modelo híbrido ResNet-18 + LSTM para classificação de violência em vídeos.

## Estrutura

```
resnet_lstm/
├── weights/          # Pesos treinados do modelo
│   └── best_model.pth
├── configs/          # Configurações de treinamento
│   └── config.json
└── experiments/      # Logs e métricas de treinamento
    └── training_history.json
```

## Dataset Utilizado

- **RWF-2000**: Dataset de classificação de violência em vídeos
  - Localização: `dataset/RWF-2000/`
  - Classes: violent, non-violent

## Treinamento

```bash
python train_pipeline.py --resnet_lstm --epochs 50 --batch_size 8
```

## Arquitetura

- **Backbone**: ResNet-18 pré-treinada no ImageNet
- **Temporal Modeling**: LSTM com 2 camadas
- **Saída**: Classificação binária (violent/non-violent)
