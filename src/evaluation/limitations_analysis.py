"""
Análise de limitações do modelo.

Identifica e analisa:
- Falsos positivos
- Falsos negativos
- Casos limítrofes
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.tensor_utils import move_to_device

from .inference import forward_batch, labels_to_numpy, unpack_batch
from .utils import create_experiment_dir, save_results

# probabilidade da classe positiva dentro da faixa considerada "limítrofe"
BORDERLINE_LOW, BORDERLINE_HIGH = 0.4, 0.6


class LimitationsAnalyzer:
    """
    Analisador de limitações do modelo.
    """

    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5,
    ):
        """
        Args:
            model: Modelo PyTorch
            dataloader: DataLoader com dados de teste
            device: Device para inferência
            threshold: Threshold para classificação
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.threshold = threshold
        self.model.eval()

    def analyze_errors(
        self,
        output_dir: str,
        experiment_name: str = "limitations",
        save_examples: bool = True,
        max_examples_per_category: int = 10,
    ) -> Dict:
        """
        Analisa erros do modelo (falsos positivos e negativos).

        Args:
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
            save_examples: Se True, salva exemplos visuais
            max_examples_per_category: Máximo de exemplos por categoria

        Returns:
            Dicionário com análise de erros
        """
        output_path = create_experiment_dir(output_dir, experiment_name)

        false_positives, false_negatives, borderline_cases = [], [], []
        total_samples = 0

        print("Analyzing model errors...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                inputs, labels = unpack_batch(batch)
                if labels is None:
                    continue
                inputs = move_to_device(inputs, self.device)

                outputs = forward_batch(self.model, inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels_to_numpy(labels)

                for i in range(len(labels_np)):
                    true_label, pred_label, prob = int(labels_np[i]), int(preds[i]), float(probs[i])
                    total_samples += 1
                    entry = {
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "probability": prob,
                        "true_label": true_label,
                        "pred_label": pred_label,
                    }
                    if pred_label == 1 and true_label == 0:
                        false_positives.append(entry)
                    elif pred_label == 0 and true_label == 1:
                        false_negatives.append(entry)
                    elif BORDERLINE_LOW <= prob <= BORDERLINE_HIGH:
                        borderline_cases.append(entry)

        results = {
            "total_samples": total_samples,
            "false_positives": _group(false_positives, max_examples_per_category),
            "false_negatives": _group(false_negatives, max_examples_per_category),
            "borderline_cases": _group(borderline_cases, max_examples_per_category),
            "error_rate": {
                "false_positive_rate": len(false_positives) / total_samples if total_samples else 0.0,
                "false_negative_rate": len(false_negatives) / total_samples if total_samples else 0.0,
                "total_error_rate": (len(false_positives) + len(false_negatives)) / total_samples if total_samples else 0.0,
            },
        }

        if save_examples:
            self._save_visual_examples(
                false_positives[:max_examples_per_category],
                false_negatives[:max_examples_per_category],
                borderline_cases[:max_examples_per_category],
                output_path,
            )

        save_results(results, output_path / "limitations_analysis.json")
        return results

    def _save_visual_examples(
        self,
        false_positives: List[Dict],
        false_negatives: List[Dict],
        borderline_cases: List[Dict],
        output_path: Path,
    ) -> None:
        """Cria a estrutura de diretórios para exemplos visuais."""
        dirs = {
            "false_positives": output_path / "false_positives",
            "false_negatives": output_path / "false_negatives",
            "borderline_cases": output_path / "borderline_cases",
        }
        for d in dirs.values():
            d.mkdir(exist_ok=True)

        print("\nVisual examples structure created in:")
        for d in dirs.values():
            print(f"  - {d}")
        print("\nNote: Actual frame extraction requires dataset access.")

    def categorize_errors(
        self,
        error_list: List[Dict],
        categories: Optional[Dict[str, callable]] = None,
    ) -> Dict:
        """
        Categoriza erros em tipos.

        Args:
            error_list: Lista de erros
            categories: Dict com funções de categorização

        Returns:
            Dicionário com erros categorizados
        """
        if categories is None:
            categories = {
                "high_confidence": lambda e: e["probability"] > 0.8,
                "medium_confidence": lambda e: 0.5 < e["probability"] <= 0.8,
                "low_confidence": lambda e: e["probability"] <= 0.5,
            }

        categorized = defaultdict(list)
        for error in error_list:
            for category, condition in categories.items():
                if condition(error):
                    categorized[category].append(error)
                    break

        return dict(categorized)

    def generate_error_report(
        self,
        output_dir: str,
        experiment_name: str = "error_report",
    ) -> Dict:
        """
        Gera relatório completo de erros.

        Args:
            output_dir: Diretório de saída
            experiment_name: Nome do experimento

        Returns:
            Dicionário com relatório
        """
        analysis = self.analyze_errors(output_dir, experiment_name)
        fp_categorized = self.categorize_errors(analysis["false_positives"]["examples"])
        fn_categorized = self.categorize_errors(analysis["false_negatives"]["examples"])

        report = {
            "summary": {
                "total_samples": analysis["total_samples"],
                "false_positives": analysis["false_positives"]["count"],
                "false_negatives": analysis["false_negatives"]["count"],
                "borderline_cases": analysis["borderline_cases"]["count"],
                "error_rates": analysis["error_rate"],
            },
            "false_positives": {
                "total": analysis["false_positives"]["count"],
                "avg_probability": analysis["false_positives"]["avg_probability"],
                "categories": {k: len(v) for k, v in fp_categorized.items()},
            },
            "false_negatives": {
                "total": analysis["false_negatives"]["count"],
                "avg_probability": analysis["false_negatives"]["avg_probability"],
                "categories": {k: len(v) for k, v in fn_categorized.items()},
            },
            "recommendations": self._generate_recommendations(analysis),
        }

        output_path = create_experiment_dir(output_dir, experiment_name)
        save_results(report, output_path / "error_report.json")
        return report

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Gera recomendações baseadas na análise."""
        fp_count = analysis["false_positives"]["count"]
        fn_count = analysis["false_negatives"]["count"]
        total = fp_count + fn_count

        if total == 0:
            return ["No errors detected. Model performance is excellent."]

        recommendations = []
        if fp_count / total > 0.6:
            recommendations.append("High false positive rate. Consider increasing threshold or improving training data.")
        if fn_count / total > 0.6:
            recommendations.append("High false negative rate. Consider decreasing threshold or adding more positive examples.")
        if analysis["false_positives"]["avg_probability"] > 0.8:
            recommendations.append("False positives have high confidence. Model may be overfitting to certain patterns.")
        if analysis["false_negatives"]["avg_probability"] < 0.3:
            recommendations.append("False negatives have low confidence. Model may need more training or better features.")

        return recommendations


def _group(examples: List[Dict], max_examples: int) -> Dict:
    """Resumo de uma categoria de erros (contagem, exemplos, prob. média)."""
    return {
        "count": len(examples),
        "examples": examples[:max_examples],
        "avg_probability": float(np.mean([e["probability"] for e in examples])) if examples else 0.0,
    }
