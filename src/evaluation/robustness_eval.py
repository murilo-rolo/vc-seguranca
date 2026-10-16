"""
Avaliação de robustez dos modelos a diferentes distorções.

Testa performance com:
- Variação de resolução
- Blur (Gaussian, Motion)
- Ruído (Gaussian, Salt & Pepper)
- Baixa iluminação (Darkening, Gamma)
- Oclusão (Random, Center)
"""

from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .inference import collect_predictions
from .metrics import calculate_metrics
from .utils import apply_distortions, create_experiment_dir, save_results


class RobustnessEvaluator:
    """
    Avaliador de robustez do modelo a distorções.
    """

    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
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
        apply_to_input: bool = True,
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
            "metrics": [],
        }

        for intensity in tqdm(intensities, desc=f"Testing {distortion_type}"):
            metrics = self._evaluate_with_distortion(distortion_type, intensity, apply_to_input)
            results["metrics"].append({"intensity": float(intensity), **metrics})

        return results

    def _evaluate_with_distortion(
        self,
        distortion_type: str,
        intensity: float,
        apply_to_input: bool,
    ) -> Dict:
        """Avalia modelo com distorção específica."""
        prepare = None
        if apply_to_input:
            def prepare(inputs):
                return self._apply_distortion_to_batch(inputs, distortion_type, intensity)

        y_true, y_pred, y_proba = collect_predictions(
            self.model, self.dataloader, self.device, prepare_inputs=prepare
        )
        return calculate_metrics(y_true, y_pred, y_proba)

    @staticmethod
    def _is_image_tensor(tensor: Any) -> bool:
        """Verifica se um tensor corresponde a dados de imagem/vídeo distortionáveis.

        Considera imagem/vídeo os tensores 5D (batch, T, C, H, W) ou 4D
        (batch, C, H, W) cuja dimensão de canal é pequena (1 ou 3). Tensores de
        outras modalidades (pose, emotion, etc.) não devem ser distorcidos.
        """
        if not isinstance(tensor, torch.Tensor):
            return False
        shape = tensor.shape
        if len(shape) == 5:
            return True
        if len(shape) == 4:
            return int(shape[1]) in (1, 3)
        return False

    def _apply_distortion_to_tensor(
        self,
        inputs: torch.Tensor,
        distortion_type: str,
        intensity: float,
    ) -> torch.Tensor:
        """Aplica distorção a um tensor de imagem (4D) ou vídeo (5D).

        Ambos os casos são reduzidos a (N, C, H, W) antes do processamento
        (vídeo: N = batch * frames), eliminando a duplicação 5D/4D.
        """
        # Tensores que não são imagem (pose, emotion, etc.) não são distorcidos
        if not self._is_image_tensor(inputs):
            return inputs

        is_video = inputs.dim() == 5
        if is_video:
            batch_size, T, C, H, W = inputs.shape
            flat = inputs.reshape(batch_size * T, C, H, W)
        else:
            flat = inputs

        n = flat.shape[0]
        frames = (flat.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # NHWC
        distorted = np.stack(
            [apply_distortions(frames[i], distortion_type, intensity) for i in range(n)]
        ).astype(np.float32) / 255.0

        distorted = torch.from_numpy(distorted).permute(0, 3, 1, 2)  # NCHW
        if is_video:
            distorted = distorted.reshape(batch_size, T, C, H, W)
        return distorted

    def _apply_distortion_to_batch(
        self,
        inputs: Any,
        distortion_type: str,
        intensity: float,
    ) -> Any:
        """Aplica distorção a um batch de inputs.

        Suporta tanto um único tensor quanto uma tupla/lista de tensores
        (ex.: modelo multimodal com (video, pose, emotion)). Apenas os tensores
        de imagem/vídeo são distorcidos; os demais são preservados.
        """
        if isinstance(inputs, (list, tuple)):
            distorted = [
                self._apply_distortion_to_tensor(x, distortion_type, intensity)
                if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            return type(inputs)(distorted)

        return self._apply_distortion_to_tensor(inputs, distortion_type, intensity)

    def evaluate_all_distortions(
        self,
        distortion_configs: Dict[str, List[float]],
        output_dir: str,
        experiment_name: str = "robustness",
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
            save_results(results, output_path / f"{distortion_type}_results.json")

        save_results(all_results, output_path / "robustness_summary.json")
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
    "center_occlusion": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}
