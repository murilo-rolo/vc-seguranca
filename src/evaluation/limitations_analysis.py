"""
Análise de limitações do modelo.

Identifica e analisa:
- Falsos positivos
- Falsos negativos
- Casos limítrofes
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import cv2
from collections import defaultdict
from tqdm import tqdm

from .metrics import calculate_metrics
from .utils import save_results, create_experiment_dir


class LimitationsAnalyzer:
    """
    Analisador de limitações do modelo.
    """
    
    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """
        Inicializa o analisador de limitações.
        
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
        max_examples_per_category: int = 10
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
        
        false_positives = []
        false_negatives = []
        borderline_cases = []
        
        all_predictions = []
        all_labels = []
        all_probs = []
        all_indices = []
        
        print("Analyzing model errors...")
        
        with torch.no_grad():
            batch_idx = 0
            for batch in tqdm(self.dataloader):
                # Obter inputs e labels
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0:-1]
                    labels = batch[-1]
                else:
                    inputs = batch
                    labels = None
                
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
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                probs_np = probs[:, 1].cpu().numpy()
                preds_np = preds.cpu().numpy()
                
                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        labels_np = labels.cpu().numpy()
                    else:
                        labels_np = np.array(labels)
                    
                    # Identificar erros
                    for i in range(len(labels_np)):
                        true_label = labels_np[i]
                        pred_label = preds_np[i]
                        prob = probs_np[i]
                        
                        all_predictions.append(pred_label)
                        all_labels.append(true_label)
                        all_probs.append(prob)
                        all_indices.append((batch_idx, i))
                        
                        # Falso positivo: predito violento, mas é não-violento
                        if pred_label == 1 and true_label == 0:
                            false_positives.append({
                                "batch_idx": batch_idx,
                                "sample_idx": i,
                                "probability": float(prob),
                                "true_label": int(true_label),
                                "pred_label": int(pred_label)
                            })
                        
                        # Falso negativo: predito não-violento, mas é violento
                        elif pred_label == 0 and true_label == 1:
                            false_negatives.append({
                                "batch_idx": batch_idx,
                                "sample_idx": i,
                                "probability": float(prob),
                                "true_label": int(true_label),
                                "pred_label": int(pred_label)
                            })
                        
                        # Caso limítrofe: probabilidade próxima ao threshold
                        elif 0.4 <= prob <= 0.6:
                            borderline_cases.append({
                                "batch_idx": batch_idx,
                                "sample_idx": i,
                                "probability": float(prob),
                                "true_label": int(true_label),
                                "pred_label": int(pred_label)
                            })
                
                batch_idx += 1
        
        # Organizar resultados
        results = {
            "false_positives": {
                "count": len(false_positives),
                "examples": false_positives[:max_examples_per_category],
                "avg_probability": float(np.mean([fp["probability"] for fp in false_positives])) if false_positives else 0.0
            },
            "false_negatives": {
                "count": len(false_negatives),
                "examples": false_negatives[:max_examples_per_category],
                "avg_probability": float(np.mean([fn["probability"] for fn in false_negatives])) if false_negatives else 0.0
            },
            "borderline_cases": {
                "count": len(borderline_cases),
                "examples": borderline_cases[:max_examples_per_category],
                "avg_probability": float(np.mean([bc["probability"] for bc in borderline_cases])) if borderline_cases else 0.0
            },
            "error_rate": {
                "false_positive_rate": len(false_positives) / len(all_labels) if all_labels else 0.0,
                "false_negative_rate": len(false_negatives) / len(all_labels) if all_labels else 0.0,
                "total_error_rate": (len(false_positives) + len(false_negatives)) / len(all_labels) if all_labels else 0.0
            }
        }
        
        # Salvar exemplos visuais se solicitado
        if save_examples:
            self._save_visual_examples(
                false_positives[:max_examples_per_category],
                false_negatives[:max_examples_per_category],
                borderline_cases[:max_examples_per_category],
                output_path
            )
        
        # Salvar resultados
        save_results(results, output_path / "limitations_analysis.json")
        
        return results
    
    def _save_visual_examples(
        self,
        false_positives: List[Dict],
        false_negatives: List[Dict],
        borderline_cases: List[Dict],
        output_path: Path
    ):
        """Salva exemplos visuais de erros."""
        # Criar diretórios
        fp_dir = output_path / "false_positives"
        fn_dir = output_path / "false_negatives"
        bc_dir = output_path / "borderline_cases"
        
        fp_dir.mkdir(exist_ok=True)
        fn_dir.mkdir(exist_ok=True)
        bc_dir.mkdir(exist_ok=True)
        
        # Salvar exemplos (nota: requer acesso aos dados originais)
        # Por enquanto, apenas criar estrutura
        print(f"\nVisual examples structure created in:")
        print(f"  - {fp_dir}")
        print(f"  - {fn_dir}")
        print(f"  - {bc_dir}")
        print("\nNote: Actual frame extraction requires dataset access.")
    
    def categorize_errors(
        self,
        error_list: List[Dict],
        categories: Optional[Dict[str, callable]] = None
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
            # Categorias padrão baseadas em probabilidade
            categories = {
                "high_confidence": lambda e: e["probability"] > 0.8,
                "medium_confidence": lambda e: 0.5 < e["probability"] <= 0.8,
                "low_confidence": lambda e: e["probability"] <= 0.5
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
        experiment_name: str = "error_report"
    ) -> Dict:
        """
        Gera relatório completo de erros.
        
        Args:
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
        
        Returns:
            Dicionário com relatório
        """
        # Analisar erros
        analysis = self.analyze_errors(output_dir, experiment_name)
        
        # Categorizar
        fp_categorized = self.categorize_errors(analysis["false_positives"]["examples"])
        fn_categorized = self.categorize_errors(analysis["false_negatives"]["examples"])
        
        # Gerar relatório
        report = {
            "summary": {
                "total_samples": len(analysis.get("all_labels", [])),
                "false_positives": analysis["false_positives"]["count"],
                "false_negatives": analysis["false_negatives"]["count"],
                "borderline_cases": analysis["borderline_cases"]["count"],
                "error_rates": analysis["error_rate"]
            },
            "false_positives": {
                "total": analysis["false_positives"]["count"],
                "avg_probability": analysis["false_positives"]["avg_probability"],
                "categories": {k: len(v) for k, v in fp_categorized.items()}
            },
            "false_negatives": {
                "total": analysis["false_negatives"]["count"],
                "avg_probability": analysis["false_negatives"]["avg_probability"],
                "categories": {k: len(v) for k, v in fn_categorized.items()}
            },
            "recommendations": self._generate_recommendations(analysis)
        }
        
        output_path = create_experiment_dir(output_dir, experiment_name)
        save_results(report, output_path / "error_report.json")
        
        return report
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Gera recomendações baseadas na análise."""
        recommendations = []
        
        fp_count = analysis["false_positives"]["count"]
        fn_count = analysis["false_negatives"]["count"]
        total = fp_count + fn_count
        
        if total == 0:
            return ["No errors detected. Model performance is excellent."]
        
        fp_rate = fp_count / total if total > 0 else 0
        fn_rate = fn_count / total if total > 0 else 0
        
        if fp_rate > 0.6:
            recommendations.append("High false positive rate. Consider increasing threshold or improving training data.")
        
        if fn_rate > 0.6:
            recommendations.append("High false negative rate. Consider decreasing threshold or adding more positive examples.")
        
        if analysis["false_positives"]["avg_probability"] > 0.8:
            recommendations.append("False positives have high confidence. Model may be overfitting to certain patterns.")
        
        if analysis["false_negatives"]["avg_probability"] < 0.3:
            recommendations.append("False negatives have low confidence. Model may need more training or better features.")
        
        return recommendations

