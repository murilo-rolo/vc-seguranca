"""
Estudo de impacto das modalidades no modelo fusionado (F5, EVAL-T3).
"""

from typing import Dict, Optional, Union

import torch

from .metrics import MetricsCalculator
from .utils import create_experiment_dir, save_results

MODALITIES = ("video", "pose", "emotion")


class ImpactStudy:
    """
    Estudo de impacto de cada modalidade no modelo fusionado.

    Em vez de trocar a arquitetura (ablação), perturba o INPUT de uma
    modalidade por vez com ruído gaussiano e mede o efeito nas logits e na
    acurácia.
    """

    def __init__(
        self,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            dataloader: DataLoader com dados de teste
            device: Device para inferência
        """
        self.dataloader = dataloader
        self.device = torch.device(device)

    def evaluate_configuration(self, model, config_name: str) -> Dict:
        """Avalia uma configuração específica (modelo) e retorna as métricas."""
        calculator = MetricsCalculator(model, self.dataloader, device=self.device)
        metrics, _, _, _ = calculator.evaluate()
        return {"config": config_name, "metrics": metrics}

    def run_impact_study(
        self,
        model,
        dataloader=None,
        noise_std: Union[float, Dict[str, float]] = 0.05,
        output_dir: Optional[str] = None,
        experiment_name: str = "impact_study",
    ) -> Dict:
        """
        Estudo de impacto de cada modalidade no modelo fusionado (F5, EVAL-T3).

        Perturba o INPUT de uma modalidade por vez com ruído gaussiano:

        - video → ruído nos frames (B, T, C, H, W)
        - pose  → ruído nos keypoints (B, T, num_joints, 3)
        - emotion → ruído nos embeddings 128-d (B, T, 128) (AD-009: não há mais
          semântica por classe, o ruído vai na stream de embeddings)

        Mede, para cada modalidade:
        - mean absolute fused-logit delta (média sobre o batch e as classes;
          também reportado por classe — "per class pair");
        - classification-accuracy drop vs o input limpo.

        Sanidade: noise_std == 0 degenera no baseline limpo (delta ≈ 0).

        Args:
            model: Modelo multimodal fusionado (forward(video, pose, emotion))
            dataloader: DataLoader multimodal (video, pose, emotion, label).
                None = usa o dataloader do próprio estudo.
            noise_std: Desvio padrão do ruído gaussiano (float para todas as
                modalidades ou dict {modalidade: std}).
            output_dir: Diretório de saída (opcional).
            experiment_name: Nome do experimento (subdiretório).

        Returns:
            Dict com deltas por modalidade e ranking por logit delta e por
            accuracy drop.
        """
        if dataloader is None:
            dataloader = self.dataloader

        if isinstance(noise_std, (int, float)):
            stds = {m: float(noise_std) for m in MODALITIES}
        else:
            stds = {m: float(noise_std.get(m, 0.05)) for m in MODALITIES}

        model.eval()
        acc = {
            m: {
                "logit_delta_sum": 0.0,
                "logit_delta_per_class_sum": None,
                "n_samples": 0,
                "n_correct_clean": 0,
                "n_correct_noisy": 0,
            }
            for m in MODALITIES
        }

        with torch.no_grad():
            for batch in dataloader:
                video, pose, emotion = batch[0], batch[1], batch[2]
                labels = batch[-1]

                video = video.to(self.device)
                pose = pose.to(self.device)
                emotion = emotion.to(self.device)
                labels = labels.to(self.device)

                logits_clean = model(video, pose, emotion)
                n = video.shape[0]
                n_correct_clean = int((logits_clean.argmax(dim=1) == labels).sum().item())
                num_classes = logits_clean.shape[1]

                for m in MODALITIES:
                    v2, p2, e2 = video.clone(), pose.clone(), emotion.clone()
                    if m == "video":
                        v2 = v2 + torch.randn_like(v2) * stds[m]
                    elif m == "pose":
                        p2 = p2 + torch.randn_like(p2) * stds[m]
                    else:
                        e2 = e2 + torch.randn_like(e2) * stds[m]

                    logits_noisy = model(v2, p2, e2)
                    delta = (logits_noisy - logits_clean).abs()

                    if acc[m]["logit_delta_per_class_sum"] is None:
                        acc[m]["logit_delta_per_class_sum"] = [0.0] * num_classes
                    per_class = delta.mean(dim=0).cpu().numpy()
                    for c in range(num_classes):
                        acc[m]["logit_delta_per_class_sum"][c] += float(per_class[c]) * n

                    acc[m]["logit_delta_sum"] += float(delta.mean().item()) * n
                    acc[m]["n_samples"] += n
                    acc[m]["n_correct_clean"] += n_correct_clean
                    acc[m]["n_correct_noisy"] += int(
                        (logits_noisy.argmax(dim=1) == labels).sum().item()
                    )

        modalities: Dict[str, Dict] = {}
        for m in MODALITIES:
            a = acc[m]
            total = max(1, a["n_samples"])
            modalities[m] = {
                "mean_abs_logit_delta": a["logit_delta_sum"] / total,
                "logit_delta_per_class": [s / total for s in a["logit_delta_per_class_sum"]],
                "accuracy_drop": (a["n_correct_clean"] - a["n_correct_noisy"]) / total,
            }

        results: Dict = {
            "method": "input_noise_perturbation",
            "noise_std": stds,
            "modalities": modalities,
            "ranking_by_logit_delta": sorted(
                MODALITIES, key=lambda m: modalities[m]["mean_abs_logit_delta"], reverse=True
            ),
            "ranking_by_accuracy_drop": sorted(
                MODALITIES, key=lambda m: modalities[m]["accuracy_drop"], reverse=True
            ),
        }

        if all(v == 0 for v in stds.values()):
            max_delta = max(modalities[m]["mean_abs_logit_delta"] for m in MODALITIES)
            results["sanity_zero_noise"] = {
                "delta_max": max_delta,
                "delta_near_zero": max_delta < 1e-9,
            }
        else:
            results["sanity_zero_noise"] = None

        if output_dir is not None:
            output_path = create_experiment_dir(output_dir, experiment_name)
            save_results(results, output_path / "impact_study.json")
            save_results(
                {k: v for k, v in results.items() if k != "modalities"},
                output_path / "impact_ranking.json",
            )
            print(f"\nEstudo de impacto salvo em: {output_path}")

        print("\n=== Ranking de impacto por modalidade ===")
        print("Por mean abs logit delta:")
        for m in results["ranking_by_logit_delta"]:
            print(f"  {m:8s} delta={modalities[m]['mean_abs_logit_delta']:.6f} "
                  f"acc_drop={modalities[m]['accuracy_drop']:.4f}")
        print("Por accuracy drop:")
        for m in results["ranking_by_accuracy_drop"]:
            print(f"  {m:8s} acc_drop={modalities[m]['accuracy_drop']:.4f} "
                  f"delta={modalities[m]['mean_abs_logit_delta']:.6f}")

        return results
