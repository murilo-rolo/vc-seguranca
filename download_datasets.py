#!/usr/bin/env python3
"""
Script unificado para download e preparação de datasets.

Este script baixa e prepara os datasets necessários para o projeto:
- RWF-2000: Dataset de violência em vídeos
- UCF101: Dataset de reconhecimento de ações (filtrado para 9 classes)
- AffectNet: Dataset de reconhecimento de emoções

Uso:
    python download_datasets.py --all              # Baixar tudo
    python download_datasets.py --rwf2000          # Baixar apenas RWF-2000
    python download_datasets.py --ucf101           # Baixar + filtrar UCF101
    python download_datasets.py --affectnet        # Baixar apenas AffectNet
    python download_datasets.py --filter-ucf101    # Apenas filtrar UCF101 existente
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

from src import paths as p


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """
    Faz download de um arquivo URL para o destino.
    
    Args:
        url: URL do arquivo
        dest: Caminho de destino
        description: Descrição do arquivo para mensagens
        
    Returns:
        True se sucesso, False caso contrário
    """
    if dest.exists():
        print(f"[SKIP] Arquivo já existe: {dest}")
        return True
    
    print(f"[DOWNLOAD] {description or dest.name}")
    print(f"  URL: {url}")
    print(f"  Destino: {dest}")
    
    try:
        def progress_hook(block_num: int, block_size: int, total_size: int):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                downloaded_mb = block_num * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progresso: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
        print()  # Nova linha após progresso
        print(f"[OK] Download concluído: {dest}")
        return True
        
    except Exception as e:
        print(f"[ERRO] Falha no download: {e}")
        if dest.exists():
            dest.unlink()  # Deletar arquivo parcial
        return False


def extract_zip(zip_path: Path, dest_dir: Path, delete_after: bool = True) -> bool:
    """
    Extrai um arquivo ZIP e opcionalmente deleta após extração.
    
    Args:
        zip_path: Caminho do arquivo ZIP
        dest_dir: Diretório de destino
        delete_after: Se True, deleta o ZIP após extração
        
    Returns:
        True se sucesso, False caso contrário
    """
    if not zip_path.exists():
        print(f"[ERRO] Arquivo ZIP não encontrado: {zip_path}")
        return False
    
    print(f"[EXTRACT] {zip_path.name} -> {dest_dir}")
    
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            zip_ref.extractall(str(dest_dir))
        
        print(f"[OK] Extração concluída: {dest_dir}")
        
        if delete_after:
            zip_path.unlink()
            print(f"[OK] ZIP deletado: {zip_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERRO] Falha na extração: {e}")
        return False


def download_rwf2000() -> bool:
    """Baixa e extrai o dataset RWF-2000."""
    print("\n" + "="*60)
    print("DOWNLOAD: RWF-2000")
    print("="*60)
    
    url = "https://www.kaggle.com/api/v1/datasets/download/vulamnguyen/rwf2000"
    zip_path = p.PROJECT_ROOT / "rwf2000.zip"
    
    if not download_file(url, zip_path, "RWF-2000 Dataset"):
        return False
    
    return extract_zip(zip_path, p.DATASET_ROOT, delete_after=True)


def download_ucf101(filter_classes: bool = True) -> bool:
    """
    Baixa e extrai o dataset UCF101.
    
    Args:
        filter_classes: Se True, filtra para 9 classes relevantes
    """
    print("\n" + "="*60)
    print("DOWNLOAD: UCF101")
    print("="*60)
    
    url = "https://www.kaggle.com/api/v1/datasets/download/matthewjansen/ucf101-action-recognition"
    zip_path = p.PROJECT_ROOT / "ucf101-action-recognition.zip"
    
    if not download_file(url, zip_path, "UCF101 Action Recognition"):
        return False
    
    if not extract_zip(zip_path, p.UCF101_ROOT, delete_after=True):
        return False
    
    if filter_classes:
        print("\n[INFO] Filtrando classes do UCF101...")
        if filter_ucf101_classes():
            print("[OK] UCF101 filtrado com sucesso!")
        else:
            print("[AVISO] Falha ao filtrar UCF101")
            return False
    
    return True


def download_affectnet() -> bool:
    """Baixa e extrai o dataset AffectNet."""
    print("\n" + "="*60)
    print("DOWNLOAD: AFFECTNET")
    print("="*60)
    
    url = "https://www.kaggle.com/api/v1/datasets/download/mstjebashazida/affectnet"
    zip_path = p.PROJECT_ROOT / "affectnet.zip"
    
    if not download_file(url, zip_path, "AffectNet Dataset"):
        return False
    
    # AffectNet tem estrutura特殊的: 'archive (3)/Train' e 'archive (3)/Test'
    if not extract_zip(zip_path, p.DATASET_ROOT, delete_after=True):
        return False
    
    # Mover para estrutura correta
    archive_dir = p.DATASET_ROOT / "archive (3)"
    if archive_dir.exists():
        affectnet_dir = p.DATASET_ROOT / "AffectNet"
        affectnet_dir.mkdir(parents=True, exist_ok=True)
        
        # Mover Train e Test
        train_src = archive_dir / "Train"
        test_src = archive_dir / "Test"
        
        if train_src.exists():
            shutil.move(str(train_src), str(affectnet_dir / "Train"))
            print(f"[OK] Movido: {train_src} -> {affectnet_dir / 'Train'}")
        
        if test_src.exists():
            shutil.move(str(test_src), str(affectnet_dir / "Test"))
            print(f"[OK] Movido: {test_src} -> {affectnet_dir / 'Test'}")
        
        # Remover diretório archive
        shutil.rmtree(str(archive_dir))
        print(f"[OK] Removido: {archive_dir}")
    
    return True


def filter_ucf101_classes() -> bool:
    """
    Filtra o dataset UCF101 mantendo apenas 9 classes relevantes.
    
    Classes a manter:
    - RELEVANTES (6): BoxingPunchingBag, BoxingSpeedBag, Fencing, Nunchucks, Punch, SumoWrestling
    - OPCIONAIS SELECIONADAS (3): Archery, CliffDiving, MilitaryParade
    
    Returns:
        True se sucesso, False caso contrário
    """
    print("\n" + "="*60)
    print("FILTRO: UCF101 (9 classes relevantes)")
    print("="*60)
    
    CLASSES_TO_KEEP = [
        "BoxingPunchingBag",
        "BoxingSpeedBag",
        "Fencing",
        "Nunchucks",
        "Punch",
        "SumoWrestling",
        "Archery",
        "CliffDiving",
        "MilitaryParade"
    ]
    
    SPLITS = ["train", "test", "val"]
    dataset_root = p.UCF101_ROOT
    
    if not dataset_root.exists():
        print(f"[ERRO] Dataset UCF101 não encontrado em: {dataset_root}")
        return False
    
    # 1. Coletar todas as classes existentes
    all_classes = set()
    for split in SPLITS:
        split_dir = dataset_root / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    all_classes.add(class_dir.name)
    
    classes_to_remove = [cls for cls in all_classes if cls not in CLASSES_TO_KEEP]
    
    print(f"\nClasses encontradas: {len(all_classes)}")
    print(f"Classes a manter: {len(CLASSES_TO_KEEP)}")
    print(f"Classes a remover: {len(classes_to_remove)}")
    
    if not classes_to_remove:
        print("[INFO] Nenhuma classe para remover (dataset já filtrado)")
        return True
    
    # 2. Filtrar arquivos CSV
    print("\n[1/3] Filtrando arquivos CSV...")
    for split in SPLITS:
        csv_path = dataset_root / f"{split}.csv"
        if csv_path.exists():
            _filter_csv_file(csv_path, CLASSES_TO_KEEP)
    
    # 3. Remover diretórios das classes não desejadas
    print("\n[2/3] Removendo diretórios das classes não desejadas...")
    removed_count = 0
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in classes_to_remove:
                try:
                    shutil.rmtree(class_dir)
                    removed_count += 1
                    print(f"[OK] Removido: {split}/{class_dir.name}")
                except Exception as e:
                    print(f"[ERRO] Erro ao remover {class_dir}: {e}")
    
    print(f"[OK] Total de diretórios removidos: {removed_count}")
    
    # 4. Validar resultado
    print("\n[3/3] Validando resultado...")
    remaining_classes = set()
    for split in SPLITS:
        split_dir = dataset_root / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    remaining_classes.add(class_dir.name)
    
    extra_classes = remaining_classes - set(CLASSES_TO_KEEP)
    missing_classes = set(CLASSES_TO_KEEP) - remaining_classes
    
    if extra_classes:
        print(f"[AVISO] Classes extras encontradas: {sorted(extra_classes)}")
    else:
        print("[OK] Nenhuma classe extra encontrada")
    
    if missing_classes:
        print(f"[AVISO] Classes esperadas não encontradas: {sorted(missing_classes)}")
    else:
        print("[OK] Todas as classes esperadas estão presentes")
    
    print(f"\nClasses mantidas ({len(remaining_classes)}):")
    for cls in sorted(remaining_classes):
        print(f"  - {cls}")
    
    return True


def _filter_csv_file(csv_path: Path, classes_to_keep: List[str]):
    """
    Filtra arquivo CSV mantendo apenas as classes desejadas.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"[AVISO] Arquivo vazio: {csv_path}")
            return
        
        header = lines[0]
        initial_count = len(lines) - 1
        filtered_lines = [header]
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                label = parts[-1].strip()
                if label in classes_to_keep:
                    filtered_lines.append(line)
        
        final_count = len(filtered_lines) - 1
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        unique_classes = set()
        for line in filtered_lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                unique_classes.add(parts[-1].strip())
        
        print(f"[OK] {csv_path.name}: {initial_count} -> {final_count} linhas ({len(unique_classes)} classes)")
        
    except Exception as e:
        print(f"[ERRO] Erro ao filtrar {csv_path}: {e}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Script unificado para download e preparação de datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Baixar tudo
  python download_datasets.py --all

  # Baixar apenas RWF-2000
  python download_datasets.py --rwf2000

  # Baixar UCF101 (sempre filtra para 9 classes)
  python download_datasets.py --ucf101

  # Baixar AffectNet
  python download_datasets.py --affectnet

  # Apenas filtrar UCF101 existente
  python download_datasets.py --filter-ucf101
        """
    )
    
    # Modos de execução
    parser.add_argument("--all", action="store_true",
                       help="Baixar todos os datasets")
    parser.add_argument("--rwf2000", action="store_true",
                       help="Baixar dataset RWF-2000")
    parser.add_argument("--ucf101", action="store_true",
                       help="Baixar dataset UCF101 (filtra automaticamente)")
    parser.add_argument("--affectnet", action="store_true",
                       help="Baixar dataset AffectNet")
    parser.add_argument("--filter-ucf101", action="store_true",
                       help="Apenas filtrar UCF101 existente (sem download)")
    
    args = parser.parse_args()
    
    # Verificar se pelo menos uma opção foi selecionada
    if not any([args.all, args.rwf2000, args.ucf101, args.affectnet, args.filter_ucf101]):
        parser.print_help()
        print("\n[ERRO] Selecione pelo menos uma opção: --all, --rwf2000, --ucf101, --affectnet, --filter-ucf101")
        return 1
    
    print("="*60)
    print("DOWNLOAD E PREPARAÇÃO DE DATASETS")
    print("="*60)
    print(f"\nDiretório de datasets: {p.DATASET_ROOT}")
    
    success = True
    
    # --all: Baixar tudo
    if args.all:
        if not download_rwf2000():
            success = False
        if not download_ucf101(filter_classes=True):
            success = False
        if not download_affectnet():
            success = False
    
    # Downloads individuais
    if args.rwf2000:
        if not download_rwf2000():
            success = False
    
    if args.ucf101:
        if not download_ucf101(filter_classes=True):
            success = False
    
    if args.affectnet:
        if not download_affectnet():
            success = False
    
    # Apenas filtrar
    if args.filter_ucf101:
        if not filter_ucf101_classes():
            success = False
    
    # Resumo
    print("\n" + "="*60)
    if success:
        print("TODAS AS OPERAÇÕES CONCLUÍDAS COM SUCESSO!")
    else:
        print("ALGUMAS OPERAÇÕES FALHARAM")
        print("Verifique os erros acima e tente novamente")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
