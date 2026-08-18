"""
Módulo para extrair frames de vídeos e salvar como tensores PyTorch.

Este script:
1. Lê vídeos de data/raw/violent e data/raw/non_violent
2. Extrai N frames uniformemente espaçados por vídeo
3. Redimensiona para tamanho fixo (112x112 ou 128x128)
4. Normaliza os frames
5. Salva como tensores .pt em data/processed/
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from src import paths as p


def extract_frames_from_video(
    video_path: Path,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (112, 112),
    normalize: bool = True
) -> Optional[torch.Tensor]:
    """
    Extrai frames de um vídeo e retorna como tensor.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        num_frames: Número de frames a extrair
        target_size: Tamanho (altura, largura) para redimensionar
        normalize: Se True, normaliza valores para [0, 1]
    
    Returns:
        Tensor de shape (num_frames, 3, H, W) ou None se erro
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return None
    
    # Obter número total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Calcular índices dos frames a extrair (uniformemente espaçados)
    if total_frames < num_frames:
        # Se o vídeo tem menos frames que o desejado, repetir frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    
    for idx in frame_indices:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # Se não conseguir ler o frame, usar o último frame válido
                if len(frames) > 0:
                    frames.append(frames[-1].clone())
                else:
                    # Se não houver frames válidos, criar frame preto
                    frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                    frame_tensor = torch.from_numpy(frame).float()
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                    if normalize:
                        frame_tensor = frame_tensor / 255.0
                    frames.append(frame_tensor)
                continue
        except Exception as e:
            # Em caso de erro, usar último frame válido ou frame preto
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                frame_tensor = torch.from_numpy(frame).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                if normalize:
                    frame_tensor = frame_tensor / 255.0
                frames.append(frame_tensor)
            continue
        
        # Converter BGR para RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        frame = cv2.resize(frame, target_size)
        
        # Converter para tensor e normalizar
        frame_tensor = torch.from_numpy(frame).float()
        
        # Transpor de (H, W, C) para (C, H, W)
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        if normalize:
            # Normalizar para [0, 1]
            frame_tensor = frame_tensor / 255.0
        
        frames.append(frame_tensor)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Empilhar frames: (num_frames, C, H, W)
    frames_tensor = torch.stack(frames)
    
    return frames_tensor


def _process_single_video(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
    target_size: Tuple[int, int],
    normalize: bool
) -> Tuple[str, bool]:
    """
    Processa um único vídeo e salva o resultado.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        output_dir: Diretório de saída para salvar tensores
        num_frames: Número de frames por vídeo
        target_size: Tamanho (altura, largura) para redimensionar
        normalize: Se True, normaliza valores para [0, 1]
    
    Returns:
        Tupla (video_name, success) onde success indica se o processamento foi bem-sucedido
    """
    try:
        # Extrair frames
        frames_tensor = extract_frames_from_video(
            video_path,
            num_frames=num_frames,
            target_size=target_size,
            normalize=normalize
        )
        
        if frames_tensor is None:
            return (video_path.name, False)
        
        # Criar pasta com nome do vídeo (sem extensão)
        video_id = video_path.stem
        video_output_dir = output_dir / video_id
        video_output_dir.mkdir(exist_ok=True)
        
        # Salvar tensor
        output_path = video_output_dir / "frame_sequence.pt"
        torch.save(frames_tensor, output_path)
        
        return (video_path.name, True)
    except Exception as e:
        return (video_path.name, False)


def process_videos(
    input_dir: Path,
    output_dir: Path,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (112, 112),
    normalize: bool = True,
    video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mov"),
    max_workers: Optional[int] = max(1, int(os.cpu_count()/4))
):
    """
    Processa todos os vídeos de um diretório e salva os frames extraídos usando processamento paralelo.
    
    Args:
        input_dir: Diretório com vídeos de entrada
        output_dir: Diretório de saída para salvar tensores
        num_frames: Número de frames por vídeo
        target_size: Tamanho (altura, largura) para redimensionar
        normalize: Se True, normaliza valores para [0, 1]
        video_extensions: Extensões de vídeo aceitas
        max_workers: Número máximo de workers paralelos. Se None, usa todos os cores disponíveis
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Listar todos os vídeos
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if len(video_files) == 0:
        print(f"Nenhum vídeo encontrado em {input_dir}")
        return
    
    print(f"Processando {len(video_files)} vídeos de {input_dir.name} usando processamento paralelo...")
    
    # Criar função parcial com parâmetros fixos
    process_func = partial(
        _process_single_video,
        output_dir=output_dir,
        num_frames=num_frames,
        target_size=target_size,
        normalize=normalize
    )
    
    # Processar vídeos em paralelo
    failed_videos = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todas as tarefas
        future_to_video = {
            executor.submit(process_func, video_path): video_path
            for video_path in video_files
        }
        
        # Processar resultados conforme completam, com barra de progresso
        for future in tqdm(as_completed(future_to_video), total=len(video_files), desc=f"Extraindo frames de {input_dir.name}"):
            video_path = future_to_video[future]
            try:
                video_name, success = future.result()
                if not success:
                    failed_videos.append(video_name)
            except Exception as e:
                failed_videos.append(video_path.name)
                print(f"Erro ao processar {video_path.name}: {e}")
    
    if failed_videos:
        print(f"\nAviso: {len(failed_videos)} vídeo(s) falharam ao processar:")
        for video_name in failed_videos:
            print(f"  - {video_name}")


def preprocess_dataset(
    raw_data_root: str = None,
    processed_data_root: str = None,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (112, 112),
    normalize: bool = True,
    max_workers: Optional[int] = None
):
    """
    Pré-processa todo o dataset: extrai frames de todos os vídeos.
    
    Args:
        raw_data_root: Raiz dos dados brutos (data/raw)
        processed_data_root: Raiz dos dados processados (data/processed)
        num_frames: Número de frames por vídeo
        target_size: Tamanho (altura, largura) para redimensionar
        normalize: Se True, normaliza valores para [0, 1]
        max_workers: Número máximo de workers paralelos. Se None, usa o padrão de process_videos
    """
    if raw_data_root is None:
        raw_data_root = str(p.RAW_DATA_ROOT)
    if processed_data_root is None:
        processed_data_root = str(p.PROCESSED_ROOT)
    raw_path = Path(raw_data_root)
    processed_path = Path(processed_data_root)
    
    # Processar vídeos violentos
    violent_input = raw_path / "violent"
    violent_output = processed_path / "violent"
    
    if violent_input.exists():
        process_videos(
            violent_input,
            violent_output,
            num_frames=num_frames,
            target_size=target_size,
            normalize=normalize,
            max_workers=max_workers
        )
    else:
        print(f"Diretório não encontrado: {violent_input}")
    
    # Processar vídeos não violentos
    non_violent_input = raw_path / "non_violent"
    non_violent_output = processed_path / "non_violent"
    
    if non_violent_input.exists():
        process_videos(
            non_violent_input,
            non_violent_output,
            num_frames=num_frames,
            target_size=target_size,
            normalize=normalize,
            max_workers=max_workers
        )
    else:
        print(f"Diretório não encontrado: {non_violent_input}")
    
    print("\nPré-processamento concluído!")


if __name__ == "__main__":
    # Executar pré-processamento
    preprocess_dataset(
        num_frames=16,
        target_size=(112, 112),
        normalize=True
    )

