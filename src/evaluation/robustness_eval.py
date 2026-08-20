"""
Avaliação de robustez dos modelos a diferentes distorções.

Testa performance com:
- Variação de resolução
- Blur (Gaussian, Motion)
- Ruído (Gaussian, Salt & Pepper)
- Baixa iluminação (Darkening, Gamma)
- Oclusão (Random, Center)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .metrics import MetricsCalculator
from .utils import apply_distortions, save_results, create_experiment_dir


class RobustnessEvaluator:
    """
    Avaliador de robustez do modelo a distorções.
    """
    
    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa o avaliador de robustez.
        
        Args:
            model: Modelo PyTorch
            dataloader: DataLoader com dados de teste
            device: Device para inferência
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.model.eval()
    
    def evaluate_distortion(
        self,
        distortion_type: str,
        intensities: List[float],
        apply_to_input: bool = True
    ) -> Dict:
        """
        Avalia modelo com diferentes intensidades de distorção.
        
        Args:
            distortion_type: Tipo de distorção
            intensities: Lista de intensidades (0.0 a 1.0)
            apply_to_input: Se True, aplica distorção aos inputs
        
        Returns:
            Dicionário com métricas para cada intensidade
        """
        results = {
            "distortion_type": distortion_type,
            "intensities": intensities,
            "metrics": []
        }
        
        for intensity in tqdm(intensities, desc=f"Testing {distortion_type}"):
            # Avaliar com esta intensidade
            metrics = self._evaluate_with_distortion(distortion_type, intensity, apply_to_input)
            results["metrics"].append({
                "intensity": float(intensity),
                **metrics
            })
        
        return results
    
    def _evaluate_with_distortion(
        self,
        distortion_type: str,
        intensity: float,
        apply_to_input: bool
    ) -> Dict:
        """Avalia modelo com distorção específica."""
        from .metrics import calculate_metrics
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Obter inputs e labels
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0:-1]
                    labels = batch[-1]
                else:
                    inputs = batch
                    labels = None
                
                # Aplicar distorção se necessário
                if apply_to_input:
                    inputs = self._apply_distortion_to_batch(inputs, distortion_type, intensity)
                
                # Mover para device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, (list, tuple)):
                    inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                # Forward pass
                if isinstance(inputs, torch.Tensor):
                    outputs = self.model(inputs)
                elif isinstance(inputs, (list, tuple)):
                    outputs = self.model(*inputs)
                else:
                    raise ValueError(f"Formato de input não suportado: {type(inputs)}")
                
                # Obter predições
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())
                
                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        all_labels.append(labels.cpu().numpy())
                    else:
                        all_labels.append(np.array(labels))
        
        # Concatenar
        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels) if len(all_labels) > 0 else None
        
        if y_true is None:
            raise ValueError("Labels não fornecidos")
        
        # Calcular métricas
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        return metrics
    
    def _apply_distortion_to_batch(
        self,
        inputs: torch.Tensor,
        distortion_type: str,
        intensity: float
    ) -> torch.Tensor:
        """Aplica distorção a um batch de inputs."""
        # Converter tensor para numpy
        if len(inputs.shape) == 5:  # (batch, T, C, H, W)
            batch_size, T, C, H, W = inputs.shape
            inputs_np = inputs.permute(0, 1, 3, 4, 2).cpu().numpy()  # (batch, T, H, W, C)
            inputs_np = (inputs_np * 255).astype(np.uint8)
            
            # Aplicar distorção a cada frame
            distorted_frames = []
            for b in range(batch_size):
                video_frames = []
                for t in range(T):
                    frame = inputs_np[b, t]
                    distorted_frame = apply_distortions(frame, distortion_type, intensity)
                    video_frames.append(distorted_frame)
                distorted_frames.append(video_frames)
            
            # Converter de volta para tensor
            distorted_array = np.array(distorted_frames)  # (batch, T, H, W, C)
            distorted_array = distorted_array.astype(np.float32) / 255.0
            distorted_tensor = torch.from_numpy(distorted_array)
            distorted_tensor = distorted_tensor.permute(0, 1, 4, 2, 3)  # (batch, T, C, H, W)
            
            return distorted_tensor
        
        elif len(inputs.shape) == 4:  # (batch, C, H, W) ou (batch, T, C, H, W) já processado
            # Similar, mas sem dimensão temporal
            batch_size, C, H, W = inputs.shape
            inputs_np = inputs.permute(0, 2, 3, 1).cpu().numpy()  # (batch, H, W, C)
            inputs_np = (inputs_np * 255).astype(np.uint8)
            
            distorted_frames = []
            for b in range(batch_size):
                frame = inputs_np[b]
                distorted_frame = apply_distortions(frame, distortion_type, intensity)
                distorted_frames.append(distorted_frame)
            
            distorted_array = np.array(distorted_frames)
            distorted_array = distorted_array.astype(np.float32) / 255.0
            distorted_tensor = torch.from_numpy(distorted_array)
            distorted_tensor = distorted_tensor.permute(0, 3, 1, 2)  # (batch, C, H, W)
            
            return distorted_tensor
        
        else:
            raise ValueError(f"Formato de input não suportado: {inputs.shape}")
    
    def evaluate_all_distortions(
        self,
        distortion_configs: Dict[str, List[float]],
        output_dir: str,
        experiment_name: str = "robustness"
    ) -> Dict:
        """
        Avalia todas as distorções configuradas.
        
        Args:
            distortion_configs: Dict {distortion_type: [intensities]}
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
        
        Returns:
            Dicionário com todos os resultados
        """
        output_path = create_experiment_dir(output_dir, experiment_name)
        all_results = {}
        
        for distortion_type, intensities in distortion_configs.items():
            print(f"\nAvaliando robustez: {distortion_type}")
            results = self.evaluate_distortion(distortion_type, intensities)
            all_results[distortion_type] = results
            
            # Salvar resultados individuais
            result_file = output_path / f"{distortion_type}_results.json"
            save_results(results, result_file)
        
        # Salvar resultados consolidados
        summary_file = output_path / "robustness_summary.json"
        save_results(all_results, summary_file)
        
        return all_results


# Configurações padrão de distorções
DEFAULT_DISTORTION_CONFIGS = {
    "resize_down": [0.0, 0.2, 0.4, 0.6, 0.8],  # Redução de resolução
    "gaussian_blur": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "motion_blur": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "gaussian_noise": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "salt_pepper_noise": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "darkening": [0.0, 0.2, 0.4, 0.6, 0.8],
    "gamma_correction": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "random_occlusion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "center_occlusion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

