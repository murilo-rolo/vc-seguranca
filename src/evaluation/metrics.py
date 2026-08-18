"""
Cálculo de métricas de avaliação para modelos de detecção de violência.

Inclui: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC, AUC-PR
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Importar modelos necessários
try:
    from src.models.multimodal_risk import MultimodalRiskDetector
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False

try:
    from src.models.cnn3d_risk import CNN3DRiskDetector
    HAS_CNN3D = True
except ImportError:
    HAS_CNN3D = False


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: List[str] = ["Non-Violent", "Violent"]
) -> Dict:
    """
    Calcula métricas de avaliação completas.
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_pred: Predições (0 ou 1)
        y_proba: Probabilidades da classe positiva (opcional, para AUC)
        class_names: Nomes das classes
    
    Returns:
        Dicionário com todas as métricas
    """
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    
    # Métricas por classe
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Métricas macro (média das classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Métricas weighted
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC e AUC-PR (se probabilidades fornecidas)
    auc_roc = None
    auc_pr = None
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
        except ValueError:
            pass
    
    # Organizar resultados
    metrics = {
        "accuracy": float(accuracy),
        "precision": {
            class_names[0]: float(precision[0]),
            class_names[1]: float(precision[1]),
            "macro": float(precision_macro),
            "weighted": float(precision_weighted)
        },
        "recall": {
            class_names[0]: float(recall[0]),
            class_names[1]: float(recall[1]),
            "macro": float(recall_macro),
            "weighted": float(recall_weighted)
        },
        "f1_score": {
            class_names[0]: float(f1[0]),
            class_names[1]: float(f1[1]),
            "macro": float(f1_macro),
            "weighted": float(f1_weighted)
        },
        "specificity": float(specificity),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "confusion_matrix_array": cm.tolist()
    }
    
    if auc_roc is not None:
        metrics["auc_roc"] = float(auc_roc)
    if auc_pr is not None:
        metrics["auc_pr"] = float(auc_pr)
    
    return metrics


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calcula métricas multiclasse (ex: 8 emoções do EmotionNet).

    Args:
        y_true: Labels verdadeiros (índices de 0 a K-1)
        y_pred: Predições (índices de 0 a K-1)
        y_proba: Matriz de probabilidades (N, K) (opcional, para AUC OvR)
        class_names: Nomes das classes (K)

    Returns:
        Dicionário com métricas (accuracy, precision/recall/f1 por classe,
        macro/weighted, confusion matrix). Mantém as mesmas chaves padrão.
    """
    n_classes = len(class_names) if class_names else int(max(y_true.max(), y_pred.max()) + 1)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": {
            **{name: float(p) for name, p in zip(class_names, precision)},
            "macro": float(precision_macro),
            "weighted": float(precision_weighted),
        },
        "recall": {
            **{name: float(r) for name, r in zip(class_names, recall)},
            "macro": float(recall_macro),
            "weighted": float(recall_weighted),
        },
        "f1_score": {
            **{name: float(f) for name, f in zip(class_names, f1)},
            "macro": float(f1_macro),
            "weighted": float(f1_weighted),
        },
        "confusion_matrix_array": cm.tolist(),
        "num_classes": n_classes,
    }

    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
        try:
            metrics["auc_roc"] = float(
                roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            )
        except ValueError:
            pass
        try:
            metrics["auc_pr"] = float(
                average_precision_score(y_true, y_proba, average='macro')
            )
        except ValueError:
            pass

    return metrics


class MetricsCalculator:
    """
    Classe para calcular e salvar métricas de avaliação.
    """
    
    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_names: List[str] = ["Non-Violent", "Violent"],
        num_classes: Optional[int] = None
    ):
        """
        Inicializa o calculador de métricas.
        
        Args:
            model: Modelo PyTorch para avaliação
            dataloader: DataLoader com dados de teste
            device: Device para inferência
            class_names: Nomes das classes
            num_classes: Número de classes. Se None, inferido de class_names
                (>2 classes = multiclass).
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.class_names = list(class_names)
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        self.model.eval()
        
        # Verificar se é modelo multimodal (o modelo fusionado é usado diretamente)
        self.is_multimodal = False
        self.use_video_model = False
        self.is_cnn3d = False
        
        if HAS_MULTIMODAL and isinstance(model, MultimodalRiskDetector):
            self.is_multimodal = True
        if HAS_CNN3D and isinstance(model, CNN3DRiskDetector):
            self.is_cnn3d = True
    
    def evaluate(self) -> Dict:
        """
        Avalia o modelo e retorna métricas.
        
        Returns:
            Dicionário com métricas
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Assumir que batch é (inputs, labels) ou similar
                # Adaptar conforme estrutura do dataloader
                if len(batch) == 2:
                    inputs = batch[0]
                    labels = batch[1]
                else:
                    # Tentar inferir estrutura
                    inputs = batch[:-1]
                    labels = batch[-1]
                
                # Mover para device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, (list, tuple)):
                    inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                # CNN3D: clipes vêm em frame-last (B, T, C, H, W) do dataloader —
                # permutar para (B, C, T, H, W), mesma semântica de
                # train_cnn3d.py:_permute_clips (o forward também auto-detecta).
                if self.is_cnn3d and isinstance(inputs, torch.Tensor):
                    if len(inputs.shape) == 5 and inputs.shape[1] != 3 and inputs.shape[2] == 3:
                        inputs = inputs.permute(0, 2, 1, 3, 4)
                
                # Forward pass padrão (o modelo fusionado é usado diretamente)
                if isinstance(inputs, torch.Tensor):
                    outputs = self.model(inputs)
                elif isinstance(inputs, (list, tuple)):
                    outputs = self.model(*inputs)
                else:
                    raise ValueError(f"Formato de input não suportado: {type(inputs)}")
                
                # Obter predições e probabilidades
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                if self.num_classes > 2:
                    # Multiclass: manter a matriz completa (N, K) para AUC OvR
                    all_probs.append(probs.cpu().numpy())
                else:
                    all_probs.append(probs[:, 1].cpu().numpy())  # Probabilidade classe positiva
                
                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        all_labels.append(labels.cpu().numpy())
                    else:
                        all_labels.append(np.array(labels))
        
        # Concatenar resultados
        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels) if len(all_labels) > 0 else None
        
        if y_true is None:
            raise ValueError("Labels não fornecidos no dataloader")
        
        # Calcular métricas
        if self.num_classes > 2:
            metrics = calculate_multiclass_metrics(y_true, y_pred, y_proba, self.class_names)
        else:
            metrics = calculate_metrics(y_true, y_pred, y_proba, self.class_names)
        
        return metrics, y_true, y_pred, y_proba
    
    def save_results(
        self,
        output_dir: str,
        experiment_name: str,
        metrics: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ):
        """
        Salva resultados da avaliação.
        
        Args:
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
            metrics: Dicionário com métricas
            y_true: Labels verdadeiros
            y_pred: Predições
            y_proba: Probabilidades
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar métricas em JSON
        metrics_file = output_path / f"{experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Salvar confusion matrix
        self._plot_confusion_matrix(
            y_true, y_pred, experiment_name, output_path / f"{experiment_name}_confusion_matrix.png"
        )
        
        # Salvar curvas ROC e PR
        if y_proba is not None:
            self._plot_roc_curve(
                y_true, y_proba, experiment_name, output_path / f"{experiment_name}_roc_curve.png"
            )
            self._plot_pr_curve(
                y_true, y_proba, experiment_name, output_path / f"{experiment_name}_pr_curve.png"
            )
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        experiment_name: str,
        output_path: Path
    ):
        """Plota e salva confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{experiment_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        experiment_name: str,
        output_path: Path
    ):
        """Plota e salva curva ROC."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{experiment_name} ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        experiment_name: str,
        output_path: Path
    ):
        """Plota e salva curva Precision-Recall."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.title(f'{experiment_name} Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

