"""
Script de treinamento para modelo de Emotion Recognition no dataset AffectNet.
"""

import argparse
import csv
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Optional, List, Tuple

from src.models.emotion_cnn import create_emotion_model
from src.models.losses import FocalLoss
from src.training.utils import run_epoch, create_dataloader
from src import paths as p

# Mapeamento de classes AffectNet
AFFECTNET_CLASSES = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'anger': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6,
    'contempt': 7
}


class AffectNetDataset(Dataset):
    """
    Dataset para AffectNet.
    
    Usa labels.csv como fonte oficial de ground truth (coluna 'label'),
    com fallback para labels por pasta caso o CSV não exista.
    
    Estrutura esperada:
    dataset/AffectNet/
    ├── labels.csv
    ├── Train/
    │   ├── neutral/
    │   ├── happy/
    │   └── ...
    └── Test/
        └── ...
    """
    
    def __init__(
        self,
        data_root: Path,
        split: str = "Train",
        transform: Optional[transforms.Compose] = None,
        min_confidence: float = 0.0
    ):
        """
        Inicializa o dataset AffectNet.
        
        Args:
            data_root: Raiz do dataset AffectNet
            split: "Train" ou "Test"
            transform: Transformações a aplicar
            min_confidence: Confiança mínima do label no CSV (0.0 = usa todos)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.min_confidence = min_confidence
        
        self.class_to_idx = AFFECTNET_CLASSES.copy()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        csv_path = self.data_root / 'labels.csv'
        if csv_path.exists():
            self.samples = self._load_from_csv(csv_path)
            source = "CSV"
        else:
            print("⚠️  labels.csv não encontrado. Usando labels por pasta (legado).")
            self.samples = self._load_from_folders()
            source = "pastas"
        
        print(f"AffectNet {split}: {len(self.samples)} amostras carregadas ({source})")
        if len(self.samples) == 0:
            print(f"  ⚠️  Nenhuma imagem encontrada em {self.data_root / split}/")
            print(f"      Verifique se o dataset foi baixado em: {self.data_root}")
        else:
            class_counts = {}
            for _, label in self.samples:
                class_counts[label] = class_counts.get(label, 0) + 1
            print(f"  Distribuição: {', '.join(f'{self.idx_to_class[k]}: {v}' for k, v in sorted(class_counts.items()))}")
    
    def _load_from_csv(self, csv_path: Path) -> List[Tuple[Path, int]]:
        samples = []
        split_dir = self.data_root / self.split
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                relFCs = float(row['relFCs'])
                if relFCs < self.min_confidence:
                    continue
                
                img_path = split_dir / row['pth']
                if not img_path.exists():
                    continue
                
                label = self.class_to_idx.get(row['label'])
                if label is None:
                    continue
                
                samples.append((img_path, label))
        
        return samples
    
    def _load_from_folders(self) -> List[Tuple[Path, int]]:
        samples = []
        split_dir = self.data_root / self.split
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                class_dir = split_dir / class_name.capitalize()
            
            if class_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    for img_path in class_dir.glob(f"*{ext}"):
                        samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(is_train: bool = True):
    if is_train:
        aug_list = [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        try:
            from torchvision.transforms import RandAugment
            aug_list.insert(0, RandAugment(num_ops=2, magnitude=9))
        except ImportError:
            pass
        aug_list += [
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(aug_list)
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])





def _plot_confusion_matrix(cm, class_names, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel='Predicted', ylabel='True'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo de Emotion Recognition no AffectNet")
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Número de épocas"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamanho do batch"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers para DataLoader"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para treinamento"
    )
    
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=8,
        help="Épocas sem melhora na val loss para early stopping"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) para o Adam"
    )
    
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.7,
        help="Confiança mínima do label no CSV (0.0 = usa todos, 0.7 recomendado)"
    )
    
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2,
        help="Gamma da Focal Loss (default: 0.5)"
    )
    
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0 = desativado, default: 1.0)"
    )
    
    parser.add_argument(
        "--sampler",
        action="store_true",
        default=True,
        help="Usar WeightedRandomSampler para balancear batches"
    )
    parser.add_argument(
        "--no-sampler",
        action="store_false",
        dest="sampler",
        help="Desabilitar WeightedRandomSampler"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar treino a partir do último best_model.pth"
    )
    
    parser.add_argument(
        "--amp",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Enable mixed precision training (default: True se CUDA disponível)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable mixed precision training"
    )
    
    args = parser.parse_args()
    
    args.dataset_path = str(p.AFFECTNET_ROOT)
    # Criar diretório de saída
    output_dir = p.EMOTION_MODELS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar datasets
    train_dataset = AffectNetDataset(
        data_root=args.dataset_path,
        split="Train",
        transform=get_transforms(is_train=True),
        min_confidence=args.min_confidence
    )
    
    val_dataset = AffectNetDataset(
        data_root=args.dataset_path,
        split="Test",
        transform=get_transforms(is_train=False),
        min_confidence=args.min_confidence
    )
    
    # Criar DataLoaders
    train_sampler = None
    if args.sampler:
        class_counts = torch.zeros(len(AFFECTNET_CLASSES))
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        sample_weights = [1.0 / class_counts[label].item() for _, label in train_dataset.samples]
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
    
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size,
        shuffle=not args.sampler, sampler=train_sampler,
        num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    
    # Criar modelo
    device = torch.device(args.device)
    
    resume_checkpoint_path = output_dir / 'best_model.pth'
    start_epoch = 1
    ckpt = None
    
    if args.resume and resume_checkpoint_path.exists():
        model, ckpt = create_emotion_model(
            num_emotions=8, pretrained=True,
            checkpoint_path=str(resume_checkpoint_path),
            resume_training=True, device=args.device
        )
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['val_acc']
        print(f"Retomando treino da época {ckpt['epoch']} (val_acc: {ckpt['val_acc']:.2f}%)")
    else:
        model = create_emotion_model(
            num_emotions=8, pretrained=True,
            device=args.device
        )
        best_val_acc = 0.0
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    criterion = FocalLoss(gamma=args.focal_gamma)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Restaurar optimizer, scheduler e early stopping no resume
    if args.resume and ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_val_acc_early = ckpt['val_acc']
        epochs_no_improve = 0
    else:
        best_val_acc_early = 0.0
        epochs_no_improve = 0
    
    # Treinamento
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    print("=" * 60)
    print("Treinamento do Modelo de Emotion Recognition")
    print("=" * 60)
    print(f"Dataset: {p.AFFECTNET_ROOT}")
    print(f"Device: {args.device}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Mixed precision: {'ON' if args.amp else 'OFF'}")
    print(f"Gradient clip:   {'ON (max_norm=' + str(args.grad_clip) + ')' if args.grad_clip > 0 else 'OFF'}")
    print(f"Sampler balanceado: {'ON' if args.sampler else 'OFF'}")
    print(f"Early stop patience: {args.early_stop_patience}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Classifier:           Linear(512→8)")
    print(f"Loss function:         Focal Loss (γ={args.focal_gamma})")
    print(f"Resume: {'ON (época ' + str(ckpt['epoch']) + ')' if args.resume and ckpt is not None else 'OFF'}")
    print("=" * 60)
    print()
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer, scaler=scaler,
            grad_clip=args.grad_clip,
            desc=f"Epoch {epoch} [Train]"
        )
        val_loss, val_acc, val_metrics = run_epoch(
            model, val_loader, criterion, device,
            is_train=False, return_metrics=True,
            desc=f"Epoch {epoch} [Val]"
        )
        
        # Atualizar learning rate (ReduceLROnPlateau baseado na val_loss)
        scheduler.step(val_loss)
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        print(f"         ↳ F1: {val_metrics['f1_score']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            
            # Salvar matriz de confusão da melhor época
            cm = np.array(val_metrics['confusion_matrix'])
            np.save(output_dir / 'confusion_matrix.npy', cm)
            _plot_confusion_matrix(
                cm,
                class_names=list(AFFECTNET_CLASSES.keys()),
                output_path=output_dir / 'confusion_matrix.png'
            )
            print(f"\n✓ Melhor modelo salvo! Val Acc: {val_acc:.2f}%")
        
        # Early stopping (baseado em val_acc)
        if val_acc > best_val_acc_early:
            best_val_acc_early = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"\n⏹ Early stopping na época {epoch} (val_acc não melhorou por {args.early_stop_patience} épocas)")
                break
        
        print()
    
    # Recarregar melhor modelo do arquivo
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Salvar histórico
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Melhor Val F1:  {max(history['val_f1']):.4f}")
    print(f"Modelo recarregado do best checkpoint (época {checkpoint['epoch']})")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

