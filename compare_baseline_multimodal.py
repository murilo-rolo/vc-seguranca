"""
Script para comparar modelo baseline (ResNet-LSTM) com modelo multimodal.

Este script:
1. Carrega ambos os modelos
2. Avalia no mesmo conjunto de teste
3. Compara métricas
4. Gera relatório de comparação
"""

import argparse
import torch
import torch.nn as nn
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

from src.models.resnet_lstm import create_model as create_baseline_model
from src.models.multimodal_risk import create_multimodal_model
from src.models.cnn3d_risk import create_cnn3d_model
from src.models.resnet_lstm import ResNetLSTM
from src.datasets.surveillance_dataset import get_dataloaders as get_baseline_dataloaders
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src import paths as p


def evaluate_model(model, dataloader, device, is_multimodal=False, video_model=None,
                   video_backbone="cnn3d"):
    """
    Avalia um modelo.
    
    Args:
        model: Modelo a avaliar
        dataloader: DataLoader de teste
        device: Device
        is_multimodal: Se True, modelo é multimodal
        video_model: Modelo de vídeo (para multimodal)
        video_backbone: Backbone de vídeo do multimodal ("cnn3d" ou "resnet_lstm")
    
    Returns:
        Dict com predições e labels
    """
    model.eval()
    if video_model:
        video_model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if is_multimodal:
                video, pose, emotion, labels = batch
                video = video.to(device)
                pose = pose.to(device)
                emotion = emotion.to(device)
                labels = labels.to(device)
                
                # Extrair features de vídeo se necessário
                if len(video.shape) == 5:  # (batch, T, C, H, W)
                    if video_backbone == "cnn3d":
                        video = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                    video_features = video_model.get_features(video)  # (batch, D_v) clip token
                else:
                    video_features = video
                
                outputs = model(video_features, pose, emotion)
            else:
                frames, labels = batch
                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(frames)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidade da classe positiva
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def calculate_metrics(results):
    """Calcula métricas de avaliação."""
    preds = results['predictions']
    labels = results['labels']
    probs = results['probabilities']
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    # AUC-ROC
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': float(accuracy),
        'precision': {
            'non_violent': float(precision[0]),
            'violent': float(precision[1])
        },
        'recall': {
            'non_violent': float(recall[0]),
            'violent': float(recall[1])
        },
        'f1_score': {
            'non_violent': float(f1[0]),
            'violent': float(f1[1])
        },
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Comparar baseline com modelo multimodal")
    
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default="models/resnet_lstm/weights/best_model.pth",
        help="Caminho para modelo baseline (ResNet-LSTM)"
    )
    parser.add_argument(
        "--multimodal_model_path",
        type=str,
        default="models/multimodal/weights/best_model.pth",
        help="Caminho para modelo multimodal"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para avaliação"
    )
    parser.add_argument(
        "--video_model_path",
        type=str,
        default=None,
        help="Checkpoint do backbone de vídeo do multimodal (padrão: conforme o video_backbone gravado no checkpoint multimodal)"
    )
    
    args = parser.parse_args()
    
    output_dir = p.COMPARISON_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Comparação: Baseline vs Multimodal")
    print("=" * 60)
    print()
    
    # Carregar modelo baseline
    print("Carregando modelo baseline...")
    baseline_model = create_baseline_model(
        num_frames=16,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        num_classes=2,
        pretrained=True,
        device=args.device
    )
    baseline_checkpoint = torch.load(args.baseline_model_path, map_location=device)
    if 'model_state_dict' in baseline_checkpoint:
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(baseline_checkpoint)
    print("✓ Modelo baseline carregado")
    
    # Carregar modelo multimodal
    print("Carregando modelo multimodal...")
    multimodal_checkpoint = torch.load(args.multimodal_model_path, map_location=device)
    fusion_method = multimodal_checkpoint.get('fusion_method', 'cross_attention')
    use_temporal = multimodal_checkpoint.get('use_temporal_modeling', True)
    # Dimensão/backbone de vídeo lidos do checkpoint (cnn3d default)
    video_backbone = multimodal_checkpoint.get('video_backbone', 'cnn3d')
    video_feature_dim = multimodal_checkpoint.get('video_feature_dim')
    if video_feature_dim is None:
        video_feature_dim = 512 if video_backbone == 'cnn3d' else 256

    multimodal_model = create_multimodal_model(
        video_feature_dim=video_feature_dim,
        pose_feature_dim=99,
        emotion_feature_dim=128,
        num_frames=16,
        fusion_method=fusion_method,
        use_temporal_modeling=use_temporal,
        device=args.device
    )
    multimodal_model.load_state_dict(multimodal_checkpoint['model_state_dict'])
    print(f"✓ Modelo multimodal carregado (backbone de vídeo: {video_backbone}, {video_feature_dim} dims)")

    # Carregar modelo de vídeo para multimodal (conforme backbone gravado no checkpoint)
    if video_backbone == "cnn3d":
        video_ckpt_path = args.video_model_path or str(p.CNN3D_WEIGHTS / "best_model.pth")
        video_ckpt = torch.load(video_ckpt_path, map_location=device)
        video_model_name = video_ckpt.get('model_name') or video_ckpt.get('backbone') or "r2plus1d_18"
        video_model = create_cnn3d_model(
            model_name=video_model_name,
            num_classes=2,
            checkpoint_path=video_ckpt_path,
            device=args.device
        )
    else:
        video_ckpt_path = args.video_model_path or str(p.RESNET_LSTM_WEIGHTS / "best_model.pth")
        video_model = create_baseline_model(
            num_frames=16,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
            num_classes=2,
            pretrained=True,
            device=args.device
        )
        video_ckpt = torch.load(video_ckpt_path, map_location=device)
        video_model.load_state_dict(video_ckpt.get('model_state_dict', video_ckpt))
    
    # Criar DataLoaders
    print("Criando DataLoaders...")
    _, _, baseline_test_loader = get_baseline_dataloaders(
        processed_data_root=str(p.PROCESSED_ROOT),
        batch_size=8,
        num_frames=16,
        num_workers=2
    )
    
    _, _, multimodal_test_loader = get_multimodal_dataloaders(
        video_data_root=str(p.PROCESSED_ROOT),
        pose_data_root=str(p.POSE_ROOT),
        emotion_data_root=str(p.EMOTION_ROOT),
        batch_size=8,
        num_frames=16,
        window_size=16,
        video_mode="frames",
        pose_mode="flatten",
        num_workers=2,
        dataset_name="rwf2000"
    )
    print("✓ DataLoaders criados")
    print()
    
    # Avaliar baseline
    print("Avaliando modelo baseline...")
    baseline_results = evaluate_model(
        baseline_model, baseline_test_loader, device, is_multimodal=False
    )
    baseline_metrics = calculate_metrics(baseline_results)
    print("✓ Baseline avaliado")
    print()
    
    # Avaliar multimodal
    print("Avaliando modelo multimodal...")
    multimodal_results = evaluate_model(
        multimodal_model, multimodal_test_loader, device,
        is_multimodal=True, video_model=video_model, video_backbone=video_backbone
    )
    multimodal_metrics = calculate_metrics(multimodal_results)
    print("✓ Multimodal avaliado")
    print()
    
    # Comparar
    print("=" * 60)
    print("Resultados da Comparação")
    print("=" * 60)
    print()
    
    print("Baseline (ResNet-LSTM):")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision (violent): {baseline_metrics['precision']['violent']:.4f}")
    print(f"  Recall (violent): {baseline_metrics['recall']['violent']:.4f}")
    print(f"  F1-Score (violent): {baseline_metrics['f1_score']['violent']:.4f}")
    print(f"  AUC-ROC: {baseline_metrics['auc_roc']:.4f}")
    print()
    
    print("Multimodal:")
    print(f"  Accuracy: {multimodal_metrics['accuracy']:.4f}")
    print(f"  Precision (violent): {multimodal_metrics['precision']['violent']:.4f}")
    print(f"  Recall (violent): {multimodal_metrics['recall']['violent']:.4f}")
    print(f"  F1-Score (violent): {multimodal_metrics['f1_score']['violent']:.4f}")
    print(f"  AUC-ROC: {multimodal_metrics['auc_roc']:.4f}")
    print()
    
    # Melhoria
    accuracy_improvement = multimodal_metrics['accuracy'] - baseline_metrics['accuracy']
    f1_improvement = multimodal_metrics['f1_score']['violent'] - baseline_metrics['f1_score']['violent']
    
    print("Melhoria:")
    print(f"  Accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)")
    print(f"  F1-Score (violent): {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
    print()
    
    # Salvar resultados
    comparison = {
        'baseline': baseline_metrics,
        'multimodal': multimodal_metrics,
        'improvement': {
            'accuracy': float(accuracy_improvement),
            'f1_score_violent': float(f1_improvement)
        },
        'fusion_method': fusion_method,
        'use_temporal_modeling': use_temporal,
        'video_backbone': video_backbone,
        'video_feature_dim': video_feature_dim
    }
    
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Resultados salvos em: {output_dir / 'comparison.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

