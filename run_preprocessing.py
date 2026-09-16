"""
Script de entrada para pré-processamento dos datasets.

Cada etapa pode ser executada individualmente com -m:
    python -m src.preprocessing.organize
    python -m src.preprocessing.frames --num_frames 16
    python -m src.preprocessing.pose --num_frames 16
    python -m src.preprocessing.emotion
    python -m src.preprocessing.index --split all
    python -m src.preprocessing.pipeline --num_frames 16
"""

from src.preprocessing.pipeline import main

if __name__ == "__main__":
    main()
