"""
Estudo de ablação para entender contribuição de cada modalidade.
"""

import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

from .metrics import MetricsCalculator
from .utils import save_results, create_experiment_dir


class AblationStudy:
    """
    Estudo de ablação para modelos multimodais.
    """
    
    def __init__(
        self,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa estudo de ablação.
        
        Args:
            dataloader: DataLoader com dados de teste
            device: Device para inferência
        """
        self.dataloader = dataloader
        self.device = torch.device(device)
    
    def evaluate_configuration(
        self,
        model,
        config_name: str
    ) -> Dict:
        """
        Avalia uma configuração específica.
        
        Args:
            model: Modelo para avaliar
            config_name: Nome da configuração
        
        Returns:
            Dicionário com métricas
        """
        calculator = MetricsCalculator(model, self.dataloader, device=self.device)
        metrics, y_true, y_pred, y_proba = calculator.evaluate()
        
        return {
            "config": config_name,
            "metrics": metrics
        }
    
    def run_ablation(
        self,
        models: Dict[str, torch.nn.Module],
        output_dir: str,
        experiment_name: str = "ablation"
    ) -> Dict:
        """
        Executa estudo de ablação completo.
        
        Args:
            models: Dict {config_name: model}
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
        
        Returns:
            Dicionário com resultados de todas as configurações
        """
        output_path = create_experiment_dir(output_dir, experiment_name)
        results = {}
        
        print("=" * 60)
        print("Running Ablation Study")
        print("=" * 60)
        
        for config_name, model in models.items():
            print(f"\nEvaluating: {config_name}")
            result = self.evaluate_configuration(model, config_name)
            results[config_name] = result
            
            # Salvar resultado individual
            save_results(result, output_path / f"{config_name}_metrics.json")
        
        # Comparação
        comparison = self._compare_configurations(results)
        results["comparison"] = comparison
        
        # Salvar comparação
        save_results(comparison, output_path / "ablation_comparison.json")
        
        # Salvar resumo completo
        save_results(results, output_path / "ablation_summary.json")
        
        return results
    
    def _compare_configurations(self, results: Dict) -> Dict:
        """Compara diferentes configurações."""
        comparison = {
            "configurations": list(results.keys()),
            "metrics_comparison": {}
        }
        
        # Extrair métricas de cada configuração
        metric_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr"]
        
        for metric_name in metric_names:
            comparison["metrics_comparison"][metric_name] = {}
            
            for config_name, result in results.items():
                metrics = result["metrics"]
                
                if metric_name in metrics:
                    if isinstance(metrics[metric_name], dict):
                        # Métricas por classe
                        comparison["metrics_comparison"][metric_name][config_name] = {
                            "macro": metrics[metric_name].get("macro"),
                            "weighted": metrics[metric_name].get("weighted")
                        }
                    else:
                        comparison["metrics_comparison"][metric_name][config_name] = metrics[metric_name]
        
        # Identificar melhor configuração por métrica
        best_configs = {}
        for metric_name, config_values in comparison["metrics_comparison"].items():
            if config_values:
                # Encontrar melhor valor
                best_config = max(config_values.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1].get("macro", 0))
                best_configs[metric_name] = {
                    "best_config": best_config[0],
                    "value": best_config[1]
                }
        
        comparison["best_configurations"] = best_configs
        
        return comparison

    def run_impact_study(
        self,
        model,
        dataloader=None,
        noise_std: Union[float, Dict[str, float]] = 0.05,
        output_dir: Optional[str] = None,
        experiment_name: str = "impact_study"
    ) -> Dict:
        """
        Estudo de impacto de cada modalidade no modelo fusionado (F5, EVAL-T3).

        Reutiliza as mecânicas de ablação, mas em vez de trocar a arquitetura,
        perturba o INPUT de uma modalidade por vez com ruído gaussiano:

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
            stds: Dict[str, float] = {
                "video": float(noise_std),
                "pose": float(noise_std),
                "emotion": float(noise_std),
            }
        else:
            stds = {m: float(noise_std.get(m, 0.05)) for m in ("video", "pose", "emotion")}

        modality_names = ["video", "pose", "emotion"]
        model.eval()

        acc = {
            m: {
                "logit_delta_sum": 0.0,
                "logit_delta_per_class_sum": None,
                "n_samples": 0,
                "n_correct_clean": 0,
                "n_correct_noisy": 0,
            }
            for m in modality_names
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
                preds_clean = logits_clean.argmax(dim=1)
                n = video.shape[0]
                n_correct_clean = int((preds_clean == labels).sum().item())
                num_classes = logits_clean.shape[1]

                for m in modality_names:
                    v2 = video.clone()
                    p2 = pose.clone()
                    e2 = emotion.clone()

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
        for m in modality_names:
            a = acc[m]
            total = max(1, a["n_samples"])
            modalities[m] = {
                "mean_abs_logit_delta": a["logit_delta_sum"] / total,
                "logit_delta_per_class": [
                    s / total for s in a["logit_delta_per_class_sum"]
                ],
                "accuracy_drop": (a["n_correct_clean"] - a["n_correct_noisy"]) / total,
            }

        results: Dict = {
            "method": "input_noise_perturbation",
            "noise_std": stds,
            "modalities": modalities,
            "ranking_by_logit_delta": sorted(
                modality_names,
                key=lambda m: modalities[m]["mean_abs_logit_delta"],
                reverse=True,
            ),
            "ranking_by_accuracy_drop": sorted(
                modality_names,
                key=lambda m: modalities[m]["accuracy_drop"],
                reverse=True,
            ),
        }

        # Sanidade: noise_std == 0 -> delta ≈ 0
        if all(v == 0 for v in stds.values()):
            max_delta = max(
                modalities[m]["mean_abs_logit_delta"] for m in modality_names
            )
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

        # Impressão do ranking (saída legível)
        print("\n=== Ranking de impacto por modalidade ===")
        print("Por mean abs logit delta:")
        for m in results["ranking_by_logit_delta"]:
            print(
                f"  {m:8s} delta={modalities[m]['mean_abs_logit_delta']:.6f} "
                f"acc_drop={modalities[m]['accuracy_drop']:.4f}"
            )
        print("Por accuracy drop:")
        for m in results["ranking_by_accuracy_drop"]:
            print(
                f"  {m:8s} acc_drop={modalities[m]['accuracy_drop']:.4f} "
                f"delta={modalities[m]['mean_abs_logit_delta']:.6f}"
            )

        return results

