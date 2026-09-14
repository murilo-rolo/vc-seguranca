"""
Script de entrada para pré-processamento dos datasets.

Cada etapa pode ser executada individualmente:
    python src/preprocessing/organize.py
    python src/preprocessing/frames.py --num_frames 16
    python src/preprocessing/pose.py --num_frames 16
    python src/preprocessing/emotion.py
    python src/preprocessing/index.py --split all
    python src/preprocessing/pipeline.py --num_frames 16
"""

from src.preprocessing.pipeline import main

if __name__ == "__main__":
    main()
