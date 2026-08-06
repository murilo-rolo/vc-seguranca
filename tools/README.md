# Tools

Esta pasta contém scripts utilitários e ferramentas auxiliares do projeto.

## Scripts Disponíveis

- **validate_data_structure.py** - Valida a estrutura de dados do projeto (datasets, dados processados, etc.)
- **compare_baseline_multimodal.py** - Compara modelos baseline com multimodal
- **train_quick_test.py** - Script para teste rápido de treinamento

## Como Usar
```

### Teste Rápido de Treinamento

```bash
python tools/train_quick_test.py
```

### Comparar Modelos

```bash
python tools/compare_baseline_multimodal.py
```

## Scripts Relacionados (Raiz do Projeto)

- **download_datasets.py** - Script unificado para download e preparação de datasets (inclui filtragem do UCF101)
- **download_datasets.sh** - Script de download original (backup, deprecated)

## Nota

Estes são scripts utilitários. Para treinamento completo, use `train_pipeline.py` na raiz do projeto.

