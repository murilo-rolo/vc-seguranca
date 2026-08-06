"""
Script Master para Orquestração Completa do Pipeline de Treinamento.

Este script automatiza todo o pipeline de treinamento do projeto:
1. Validação da estrutura de dados
2. Treinamento de modelos base (ResNet-LSTM, EmotionNet, CNN 3D)
3. Treinamento do modelo multimodal
4. Avaliação final

Uso:
    # Treinar tudo do zero
    python train_pipeline.py --all
    
    # Treinar apenas modelos base
    python train_pipeline.py --base_models
    
    # Treinar apenas multimodal (assumindo que modelos base já existem)
    python train_pipeline.py --multimodal
    
    # Treinar com opções customizadas
    python train_pipeline.py --all --skip_emotion --skip_cnn3d
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import os

from src import paths as p


class TrainingPipeline:
    """Orquestrador do pipeline de treinamento."""
    
    def __init__(self, project_root: str = None, force_retrain: bool = False):
        if project_root is None:
            project_root = str(p.PROJECT_ROOT)
        self.project_root = Path(project_root)
        self.steps_completed = []
        self.steps_failed = []
        self.force_retrain = force_retrain
    
    def run_command(self, cmd: List[str], description: str, required: bool = True) -> bool:
        """
        Executa um comando e retorna True se sucesso.
        
        Args:
            cmd: Lista de strings do comando
            description: Descrição do passo
            required: Se True, falha interrompe o pipeline
        
        Returns:
            True se sucesso, False caso contrário
        """
        print("\n" + "=" * 70)
        print(f"EXECUTANDO: {description}")
        print("=" * 70)
        print(f"Comando: {' '.join(cmd)}")
        print("-" * 70)
        
        try:
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            self.steps_completed.append(description)
            print(f"\n✅ {description} concluído com sucesso!")
            return True
        except subprocess.CalledProcessError as e:
            self.steps_failed.append(description)
            print(f"\n❌ {description} falhou com código {e.returncode}")
            if required:
                print(f"❌ Pipeline interrompido (etapa obrigatória falhou)")
                return False
            else:
                print(f"⚠️  Continuando pipeline (etapa opcional)")
                return False
        except Exception as e:
            self.steps_failed.append(description)
            print(f"\n❌ Erro ao executar {description}: {str(e)}")
            if required:
                return False
            return False
    
    def check_prerequisites(self) -> bool:
        """Verifica pré-requisitos antes de iniciar."""
        print("\n" + "=" * 70)
        print("VERIFICANDO PRÉ-REQUISITOS")
        print("=" * 70)
        
        checks = []
        
        # Verificar se datasets existem
        rwf_path = p.RWF2000_ROOT
        if rwf_path.exists():
            print("✅ Dataset RWF-2000 encontrado")
            checks.append(True)
        else:
            print("❌ Dataset RWF-2000 não encontrado (obrigatório)")
            checks.append(False)
        
        # Verificar se dados processados existem
        processed_path = p.PROCESSED_ROOT
        if processed_path.exists():
            print("✅ Dados processados encontrados")
            checks.append(True)
        else:
            print("⚠️  Dados processados não encontrados (execute pré-processamento primeiro)")
            checks.append(False)
        
        # Verificar se pose existe
        pose_path = p.POSE_ROOT / "rwf2000"
        if pose_path.exists():
            print("✅ Dados de pose encontrados")
            checks.append(True)
        else:
            print("⚠️  Dados de pose não encontrados (necessário para multimodal)")
            checks.append(False)
        
        # Verificar se emoção existe
        emotion_path = p.EMOTION_ROOT / "rwf2000"
        if emotion_path.exists():
            print("✅ Dados de emoção encontrados")
            checks.append(True)
        else:
            print("⚠️  Dados de emoção não encontrados (necessário para multimodal)")
            checks.append(False)
        
        all_ok = all(checks)
        if not all_ok:
            print("\n⚠️  Alguns pré-requisitos estão faltando. O pipeline pode falhar.")
            try:
                response = input("Deseja continuar mesmo assim? (s/n): ").lower()
                if response != 's':
                    return False
            except (EOFError, KeyboardInterrupt):
                # Ambiente não interativo - continuar automaticamente
                print("Ambiente não interativo detectado. Continuando automaticamente...")
        
        return True
    
    def train_resnet_lstm(self, epochs: int = 50, batch_size: int = 8, **kwargs) -> bool:
        """Treina modelo ResNet-LSTM."""
        model_path = p.RESNET_LSTM_WEIGHTS / "best_model.pth"
        
        # Verificar se já existe
        if model_path.exists() and not self.force_retrain:
            print(f"\n⚠️  Modelo ResNet-LSTM já existe em {model_path}")
            try:
                response = input("Deseja treinar novamente? (s/n): ").lower()
                if response != 's':
                    print("⏭️  Pulando treinamento de ResNet-LSTM")
                    return True
            except (EOFError, KeyboardInterrupt):
                print("Ambiente não interativo. Pulando treinamento (use --force_retrain para forçar)")
                return True
        
        cmd = [
            sys.executable, "-m", "src.training.train",
            "--num_epochs", str(epochs),
            "--batch_size", str(batch_size),
        ]
        
        # Adicionar argumentos opcionais
        if "learning_rate" in kwargs:
            cmd.extend(["--learning_rate", str(kwargs["learning_rate"])])
        if "hidden_size" in kwargs:
            cmd.extend(["--hidden_size", str(kwargs["hidden_size"])])
        if "num_layers" in kwargs:
            cmd.extend(["--num_layers", str(kwargs["num_layers"])])
        if "dropout" in kwargs:
            cmd.extend(["--dropout", str(kwargs["dropout"])])
        if "device" in kwargs:
            cmd.extend(["--device", kwargs["device"]])
        
        return self.run_command(cmd, "Treinamento ResNet-LSTM", required=True)
    
    def train_emotion_net(self, dataset_path: str, epochs: int = 50, batch_size: int = 32, **kwargs) -> bool:
        """Treina modelo EmotionNet."""
        model_path = p.EMOTION_CNN_WEIGHTS / "best_model.pth"
        
        # Verificar se já existe
        if model_path.exists() and not self.force_retrain:
            print(f"\n⚠️  Modelo EmotionNet já existe em {model_path}")
            try:
                response = input("Deseja treinar novamente? (s/n): ").lower()
                if response != 's':
                    print("⏭️  Pulando treinamento de EmotionNet")
                    return True
            except (EOFError, KeyboardInterrupt):
                print("Ambiente não interativo. Pulando treinamento (use --force_retrain para forçar)")
                return True
        
        affectnet_path = Path(dataset_path)
        if not affectnet_path.exists():
            print(f"⚠️  Dataset AffectNet não encontrado em {dataset_path}")
            print("⏭️  Pulando treinamento de EmotionNet")
            return False
        
        cmd = [
            sys.executable, "train_emotion_model.py",
            "--dataset_path", str(affectnet_path),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
        ]
        
        if "learning_rate" in kwargs:
            cmd.extend(["--learning_rate", str(kwargs["learning_rate"])])
        if "device" in kwargs:
            cmd.extend(["--device", kwargs["device"]])
        
        return self.run_command(cmd, "Treinamento EmotionNet", required=False)
    
    def train_cnn3d(self, stage: str, dataset: str, epochs: int = 50, pretrained_path: Optional[str] = None, **kwargs) -> bool:
        """Treina modelo CNN 3D."""
        if stage == "pretrain":
            model_path = p.CNN3D_UCF101_WEIGHTS / "best_model.pth"
        else:
            model_path = p.CNN3D_RWF2000_WEIGHTS / "best_model.pth"
        
        # Verificar se já existe
        if model_path.exists() and not self.force_retrain:
            print(f"\n⚠️  Modelo CNN 3D ({stage}) já existe em {model_path}")
            try:
                response = input("Deseja treinar novamente? (s/n): ").lower()
                if response != 's':
                    print(f"⏭️  Pulando treinamento de CNN 3D ({stage})")
                    return True
            except (EOFError, KeyboardInterrupt):
                print("Ambiente não interativo. Pulando treinamento (use --force_retrain para forçar)")
                return True
        
        cmd = [
            sys.executable, "train_cnn3d.py",
            "--stage", stage,
            "--dataset", dataset,
            "--epochs", str(epochs),
        ]
        
        if pretrained_path:
            cmd.extend(["--pretrained_path", pretrained_path])
        
        if "model_name" in kwargs:
            cmd.extend(["--model_name", kwargs["model_name"]])
        if "batch_size" in kwargs:
            cmd.extend(["--batch_size", str(kwargs["batch_size"])])
        if "device" in kwargs:
            cmd.extend(["--device", kwargs["device"]])
        
        return self.run_command(cmd, f"Treinamento CNN 3D ({stage})", required=False)
    
    def train_multimodal(self, epochs: int = 50, fusion_method: str = "late", **kwargs) -> bool:
        """Treina modelo multimodal."""
        model_path = p.MULTIMODAL_WEIGHTS / "best_model.pth"
        
        # Verificar se já existe
        if model_path.exists() and not self.force_retrain:
            print(f"\n⚠️  Modelo Multimodal já existe em {model_path}")
            try:
                response = input("Deseja treinar novamente? (s/n): ").lower()
                if response != 's':
                    print("⏭️  Pulando treinamento Multimodal")
                    return True
            except (EOFError, KeyboardInterrupt):
                print("Ambiente não interativo. Pulando treinamento (use --force_retrain para forçar)")
                return True
        
        # Verificar modelos base
        video_model_path = p.RESNET_LSTM_WEIGHTS / "best_model.pth"
        emotion_model_path = p.EMOTION_CNN_WEIGHTS / "best_model.pth"
        
        if not video_model_path.exists():
            print(f"❌ Modelo ResNet-LSTM não encontrado em {video_model_path}")
            print("   Execute treinamento de ResNet-LSTM primeiro")
            return False
        
        cmd = [
            sys.executable, "train_multimodal.py",
            "--epochs", str(epochs),
            "--fusion_method", fusion_method,
            "--video_model_path", str(video_model_path),
        ]
        
        if emotion_model_path.exists():
            cmd.extend(["--emotion_model_path", str(emotion_model_path)])
        else:
            print("⚠️  Modelo EmotionNet não encontrado. Continuando sem ele.")
        
        if "batch_size" in kwargs:
            cmd.extend(["--batch_size", str(kwargs["batch_size"])])
        if "use_temporal_modeling" in kwargs and kwargs["use_temporal_modeling"]:
            cmd.append("--use_temporal_modeling")
        if "device" in kwargs:
            cmd.extend(["--device", kwargs["device"]])
        
        return self.run_command(cmd, "Treinamento Multimodal", required=True)
    
    def run_full_pipeline(
        self,
        skip_emotion: bool = False,
        skip_cnn3d: bool = False,
        **kwargs
    ):
        """Executa pipeline completo de treinamento."""
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETO DE TREINAMENTO")
        print("=" * 70)
        
        # Verificar pré-requisitos
        if not self.check_prerequisites():
            print("\n❌ Pré-requisitos não atendidos. Pipeline cancelado.")
            return False
        
        success = True
        
        # 1. Treinar ResNet-LSTM
        if not self.train_resnet_lstm(**kwargs):
            success = False
            return success
        
        # 2. Treinar EmotionNet (opcional)
        if not skip_emotion:
            affectnet_path = kwargs.get("affectnet_path", "dataset/AffectNet")
            if not self.train_emotion_net(affectnet_path, **kwargs):
                print("⚠️  Falha no treinamento de EmotionNet (continuando)")
        else:
            print("\n⏭️  Pulando treinamento de EmotionNet (--skip_emotion)")
        
        # 3. Treinar CNN 3D (opcional)
        if not skip_cnn3d:
            # Pré-treinamento em UCF101
            ucf101_path = kwargs.get("ucf101_path", "dataset/UCF101")
            if Path(ucf101_path).exists():
                if not self.train_cnn3d("pretrain", "ucf101", **kwargs):
                    print("⚠️  Falha no pré-treinamento de CNN 3D (continuando)")
                
                # Fine-tuning em RWF-2000
                pretrained_path = str(p.CNN3D_UCF101_WEIGHTS / "best_model.pth")
                if Path(pretrained_path).exists():
                    if not self.train_cnn3d("finetune", "rwf2000", pretrained_path=pretrained_path, **kwargs):
                        print("⚠️  Falha no fine-tuning de CNN 3D (continuando)")
            else:
                print(f"\n⏭️  Dataset UCF101 não encontrado em {ucf101_path}")
                print("   Pulando treinamento de CNN 3D")
        else:
            print("\n⏭️  Pulando treinamento de CNN 3D (--skip_cnn3d)")
        
        # 4. Treinar Multimodal
        if not self.train_multimodal(**kwargs):
            success = False
            return success
        
        # Resumo
        print("\n" + "=" * 70)
        print("RESUMO DO PIPELINE")
        print("=" * 70)
        print(f"\n✅ Etapas concluídas: {len(self.steps_completed)}")
        for step in self.steps_completed:
            print(f"   - {step}")
        
        if self.steps_failed:
            print(f"\n❌ Etapas com falha: {len(self.steps_failed)}")
            for step in self.steps_failed:
                print(f"   - {step}")
        
        if success:
            print("\n✅ Pipeline completo executado com sucesso!")
            print("\nPróximos passos:")
            print("1. Avaliar modelos: python run_evaluation.py")
            print("2. Testar em tempo real: python run_realtime_risk_detection.py")
        else:
            print("\n⚠️  Pipeline concluído com algumas falhas. Verifique os erros acima.")
        
        return success


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Script Master para Orquestração do Pipeline de Treinamento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Treinar tudo do zero
  python train_pipeline.py --all

  # Treinar apenas modelos base
  python train_pipeline.py --base_models

  # Treinar apenas multimodal (assumindo modelos base já existem)
  python train_pipeline.py --multimodal

  # Treinar com opções customizadas
  python train_pipeline.py --all --skip_emotion --skip_cnn3d --epochs 30

  # Treinar apenas ResNet-LSTM
  python train_pipeline.py --resnet_lstm --epochs 50 --batch_size 8
        """
    )
    
    # Modos de execução
    parser.add_argument("--all", action="store_true",
                       help="Executar pipeline completo")
    parser.add_argument("--base_models", action="store_true",
                       help="Treinar apenas modelos base (ResNet-LSTM, EmotionNet, CNN 3D)")
    parser.add_argument("--multimodal", action="store_true",
                       help="Treinar apenas modelo multimodal")
    parser.add_argument("--resnet_lstm", action="store_true",
                       help="Treinar apenas ResNet-LSTM")
    parser.add_argument("--emotion", action="store_true",
                       help="Treinar apenas EmotionNet")
    parser.add_argument("--cnn3d", action="store_true",
                       help="Treinar apenas CNN 3D")
    
    # Opções de pular etapas
    parser.add_argument("--skip_emotion", action="store_true",
                       help="Pular treinamento de EmotionNet")
    parser.add_argument("--skip_cnn3d", action="store_true",
                       help="Pular treinamento de CNN 3D")
    parser.add_argument("--force_retrain", action="store_true",
                       help="Forçar retreinamento mesmo se modelo já existir")
    
    # Parâmetros de treinamento
    parser.add_argument("--epochs", type=int, default=50,
                       help="Número de épocas (padrão: 50)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Tamanho do batch (padrão: 8)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Taxa de aprendizado (padrão: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device para treinamento (padrão: cuda)")
    parser.add_argument("--fusion_method", type=str, default="late",
                       choices=["early", "late", "attention"],
                       help="Método de fusão para multimodal (padrão: late)")
    
        # Caminhos de datasets
    parser.add_argument("--affectnet_path", type=str, default=None,
                       help="Caminho para dataset AffectNet")
    parser.add_argument("--ucf101_path", type=str, default=None,
                       help="Caminho para dataset UCF101")
    parser.add_argument("--project_root", type=str, default=None,
                       help="Diretório raiz do projeto")
    
    args = parser.parse_args()
    if args.affectnet_path is None:
        args.affectnet_path = str(p.AFFECTNET_ROOT)
    if args.ucf101_path is None:
        args.ucf101_path = str(p.UCF101_ROOT)
    if args.project_root is None:
        args.project_root = str(p.PROJECT_ROOT)
    
    # Criar pipeline
    pipeline = TrainingPipeline(project_root=args.project_root, force_retrain=args.force_retrain)
    
    # Preparar kwargs
    kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "fusion_method": args.fusion_method,
        "affectnet_path": args.affectnet_path,
        "ucf101_path": args.ucf101_path,
    }
    
    # Executar modo selecionado
    if args.all:
        pipeline.run_full_pipeline(
            skip_emotion=args.skip_emotion,
            skip_cnn3d=args.skip_cnn3d,
            **kwargs
        )
    elif args.base_models:
        pipeline.train_resnet_lstm(**kwargs)
        if not args.skip_emotion:
            pipeline.train_emotion_net(args.affectnet_path, **kwargs)
        if not args.skip_cnn3d:
            if Path(args.ucf101_path).exists():
                pipeline.train_cnn3d("pretrain", "ucf101", **kwargs)
                pretrained_path = str(p.CNN3D_UCF101_WEIGHTS / "best_model.pth")
                if Path(pretrained_path).exists():
                    pipeline.train_cnn3d("finetune", "rwf2000", pretrained_path=pretrained_path, **kwargs)
    elif args.multimodal:
        pipeline.train_multimodal(**kwargs)
    elif args.resnet_lstm:
        pipeline.train_resnet_lstm(**kwargs)
    elif args.emotion:
        pipeline.train_emotion_net(args.affectnet_path, **kwargs)
    elif args.cnn3d:
        if Path(args.ucf101_path).exists():
            pipeline.train_cnn3d("pretrain", "ucf101", **kwargs)
            pretrained_path = str(p.CNN3D_UCF101_WEIGHTS / "best_model.pth")
            if Path(pretrained_path).exists():
                pipeline.train_cnn3d("finetune", "rwf2000", pretrained_path=pretrained_path, **kwargs)
    else:
        parser.print_help()
        print("\n❌ Nenhum modo de execução especificado. Use --all, --base_models, --multimodal, etc.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

