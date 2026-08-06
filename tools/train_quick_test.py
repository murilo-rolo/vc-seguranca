"""
Script rápido para testar o treinamento com poucas épocas.

Ideal para:
- Testar se o pipeline está funcionando
- Computadores com recursos limitados
- Validação rápida antes de treinar por completo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import train
from src import paths as p

if __name__ == "__main__":
    print("="*60)
    print("TREINAMENTO RÁPIDO - TESTE (10 ÉPOCAS)")
    print("="*60)
    print("\nConfiguração otimizada para computadores com recursos limitados:")
    print("  - 10 épocas")
    print("  - Batch size: 4 (reduzido para economizar memória)")
    print("  - Early stopping: 5 épocas")
    print("  - Learning rate scheduler: ativado")
    print("\n" + "="*60 + "\n")
    
    train(
        batch_size=4,  # Reduzido para economizar memória
        num_frames=16,
        num_epochs=10,  # Apenas 10 épocas para teste
        learning_rate=1e-4,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        num_workers=2,  # Reduzido para economizar recursos
        device=None,  # Auto-detecta (CPU ou GPU)
        save_dir=str(p.RESNET_LSTM_WEIGHTS),
        seed=42,
        early_stopping_patience=5,  # Para mais cedo se não melhorar
        use_scheduler=True
    )
    
    print("\n" + "="*60)
    print("TESTE CONCLUÍDO!")
    print("="*60)
    print("\nSe o teste funcionou, você pode treinar por mais épocas usando:")
    print("  python -m src.training.train --num_epochs 50 --batch_size 8")
    print("="*60)

