"""
Funções auxiliares para avaliação experimental.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict


def apply_resize(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """Redimensiona frame."""
    return cv2.resize(frame, target_size)


def apply_gaussian_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    """Aplica blur gaussiano."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_motion_blur(frame: np.ndarray, angle: float, length: int) -> np.ndarray:
    """Aplica blur de movimento."""
    # Criar kernel de movimento
    kernel = np.zeros((length, length))
    kernel[int((length-1)/2), :] = np.ones(length)
    kernel = kernel / length
    
    # Rotacionar kernel
    M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    
    # Aplicar blur
    return cv2.filter2D(frame, -1, kernel)


def apply_gaussian_noise(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Adiciona ruído gaussiano."""
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(frame: np.ndarray, density: float) -> np.ndarray:
    """Adiciona ruído salt & pepper."""
    noisy = frame.copy()
    num_salt = np.ceil(density * frame.size * 0.5)
    num_pepper = np.ceil(density * frame.size * 0.5)
    
    # Salt
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in frame.shape]
    noisy[coords[0], coords[1], :] = 255
    
    # Pepper
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in frame.shape]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy


def apply_darkening(frame: np.ndarray, factor: float) -> np.ndarray:
    """Escurece frame multiplicando por factor."""
    darkened = (frame.astype(np.float32) * factor).astype(np.uint8)
    return darkened


def apply_gamma_correction(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Aplica correção gamma."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_random_occlusion(frame: np.ndarray, occlusion_ratio: float) -> np.ndarray:
    """Aplica oclusão aleatória com boxes."""
    occluded = frame.copy()
    h, w = frame.shape[:2]
    
    # Calcular área de oclusão
    total_area = h * w
    occlusion_area = int(total_area * occlusion_ratio)
    
    # Criar múltiplas boxes aleatórias
    num_boxes = max(1, int(occlusion_ratio * 10))
    box_area = occlusion_area // num_boxes
    
    for _ in range(num_boxes):
        box_size = int(np.sqrt(box_area))
        x = np.random.randint(0, w - box_size)
        y = np.random.randint(0, h - box_size)
        occluded[y:y+box_size, x:x+box_size] = 0
    
    return occluded


def apply_center_occlusion(frame: np.ndarray, occlusion_ratio: float) -> np.ndarray:
    """Aplica oclusão no centro da imagem."""
    occluded = frame.copy()
    h, w = frame.shape[:2]
    
    # Calcular tamanho da oclusão
    occl_size = int(np.sqrt(h * w * occlusion_ratio))
    
    # Centralizar
    x = (w - occl_size) // 2
    y = (h - occl_size) // 2
    
    occluded[y:y+occl_size, x:x+occl_size] = 0
    
    return occluded


def apply_distortions(
    frame: np.ndarray,
    distortion_type: str,
    intensity: float
) -> np.ndarray:
    """
    Aplica distorção ao frame.
    
    Args:
        frame: Frame RGB (H, W, C)
        distortion_type: Tipo de distorção
        intensity: Intensidade da distorção
    
    Returns:
        Frame distorcido
    """
    if distortion_type == "resize_down":
        size = int(224 * (1 - intensity))
        size = max(64, size)  # Mínimo 64x64
        return apply_resize(frame, (size, size))
    
    elif distortion_type == "gaussian_blur":
        kernel_size = int(3 + intensity * 10)
        return apply_gaussian_blur(frame, kernel_size)
    
    elif distortion_type == "motion_blur":
        angle = intensity * 180
        length = int(5 + intensity * 20)
        return apply_motion_blur(frame, angle, length)
    
    elif distortion_type == "gaussian_noise":
        sigma = intensity * 25
        return apply_gaussian_noise(frame, sigma)
    
    elif distortion_type == "salt_pepper_noise":
        density = intensity * 0.1
        return apply_salt_pepper_noise(frame, density)
    
    elif distortion_type == "darkening":
        factor = 1.0 - intensity
        return apply_darkening(frame, factor)
    
    elif distortion_type == "gamma_correction":
        gamma = 0.5 + intensity * 0.5  # 0.5 a 1.0
        return apply_gamma_correction(frame, gamma)
    
    elif distortion_type == "random_occlusion":
        return apply_random_occlusion(frame, intensity)
    
    elif distortion_type == "center_occlusion":
        return apply_center_occlusion(frame, intensity)
    
    else:
        raise ValueError(f"Tipo de distorção não suportado: {distortion_type}")


def save_results(results: Dict, output_path: Path) -> None:
    """Salva um dicionário de resultados como JSON (cria diretórios)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(input_path: Path) -> Dict:
    """Carrega resultados de arquivo JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Cria diretório para experimento."""
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

