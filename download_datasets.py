#!/usr/bin/env python3
"""
Script unificado para download e preparação de datasets.

Este script baixa e prepara os datasets necessários para o projeto:
- RWF-2000: Dataset de violência em vídeos
- AffectNet (balanced): Dataset de reconhecimento de emoções (dollyprajapati182/balanced-affectnet)

Uso:
    python download_datasets.py --all              # Baixar tudo
    python download_datasets.py --rwf2000          # Baixar apenas RWF-2000
    python download_datasets.py --affectnet        # Baixar apenas AffectNet
"""

import argparse
import os
import shutil
import sys
import subprocess
import zipfile
from pathlib import Path, PurePath
from typing import List, Optional

from src import paths as p


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """
    Faz download de um arquivo usando aria2c (16 conexões paralelas).

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

    if not shutil.which("aria2c"):
        print("[ERRO] aria2c não encontrado. Instale com:")
        print("  sudo apt install aria2          # Debian/Ubuntu")
        print("  sudo pacman -S aria2            # Arch")
        print("  brew install aria2              # macOS")
        print("  choco install aria2             # Windows (Chocolatey)")
        return False

    print(f"[DOWNLOAD] {description or dest.name}")
    print(f"  URL: {url}")
    print(f"  Destino: {dest}")
    print(f"  Motor: aria2c (16 conexões paralelas)")

    cmd = [
        "aria2c",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--continue=true",
        "--max-tries=5",
        "--retry-wait=3",
        "--timeout=60",
        "--connect-timeout=30",
        "--auto-file-renaming=false",
        "--console-log-level=notice",
        "--summary-interval=0",
        "-d", str(dest.parent),
        "-o", dest.name,
        url,
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0 and dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[OK] Download concluído: {dest} ({size_mb:.1f} MB)")
        return True

    print(f"[ERRO] aria2c falhou (código {result.returncode})")
    if dest.exists():
        dest.unlink()
    return False


def _decode_zip_name(info: zipfile.ZipInfo) -> str:
    """
    Decodifica o nome de um membro do ZIP de forma robusta.

    O zipfile do Python decodifica nomes sem o flag UTF-8 como CP437.
    Isso gera 'mojibake' (e nomes longos demais) para arquivos gravados
    em UTF-8 sem o flag, como os vídeos em cirílico do RWF-2000.
    Aqui tentamos recuperar o UTF-8 original quando o flag não está presente.
    """
    if info.flag_bits & 0x800:
        return info.filename

    raw = getattr(info, "orig_filename", info.filename)
    if isinstance(raw, bytes):
        # Python < 3.11: orig_filename guarda os bytes crus do arquivo
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return info.filename

    # Python >= 3.11: orig_filename é a string decodificada como CP437.
    # Revertemos a codificação para obter os bytes originais e tentamos UTF-8.
    try:
        return raw.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def _sanitize_component(component: str, max_name_len: int = 255) -> str:
    """
    Trunca um componente de caminho para caber no limite de bytes do filesystem.

    A truncagem é feita por bytes (sem cortar no meio de um caractere
    multibyte) e preserva a extensão do arquivo.
    """
    encoded = component.encode("utf-8")
    if len(encoded) <= max_name_len:
        return component

    dot = component.rfind(".")
    if dot > 0:
        stem, suffix = component[:dot], component[dot:]
    else:
        stem, suffix = component, ""

    available = max_name_len - len(suffix.encode("utf-8"))
    if available <= 0:
        return component.encode("utf-8")[:max_name_len].decode("utf-8", errors="ignore")

    stem_bytes = stem.encode("utf-8")
    truncated = stem_bytes[:available].decode("utf-8", errors="ignore")
    return truncated + suffix


def _unique_path(path: Path, max_name_len: int = 255) -> Path:
    """Garante um caminho sem colisão, adicionando sufixo '_N' se necessário."""
    if not path.exists():
        return path
    suffix = path.suffix
    suffix_bytes = len(suffix.encode("utf-8"))
    counter = 1
    while True:
        disambig = f"_{counter}"
        stem_limit = max_name_len - suffix_bytes - len(disambig.encode("utf-8"))
        stem = _sanitize_component(path.stem, stem_limit)
        candidate = path.with_name(stem + disambig + suffix)
        if not candidate.exists():
            return candidate
        counter += 1


def _extract_member(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo, dest_dir: Path, max_name_len: int = 255) -> Path:
    """
    Extrai um membro do ZIP de forma segura:
    - corrige a codificação do nome (UTF-8 vs CP437)
    - evita path traversal ('..' e caminhos absolutos)
    - trunca componentes que excedam o limite do filesystem
    """
    name = _decode_zip_name(info).replace("\\", "/")
    relative = PurePath(name)
    parts = [
        _sanitize_component(part, max_name_len)
        for part in relative.parts
        if part not in ("", ".", "..")
    ]
    if not parts:
        return dest_dir

    target = dest_dir
    for part in parts[:-1]:
        target = target / part

    final = target / parts[-1]
    if info.is_dir():
        final.mkdir(parents=True, exist_ok=True)
        return final

    target.mkdir(parents=True, exist_ok=True)
    final = _unique_path(final, max_name_len)
    with zip_ref.open(info) as src, open(final, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return final


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
        
        try:
            max_name_len = os.pathconf(str(dest_dir), "PC_NAME_MAX")
        except (OSError, ValueError, AttributeError):
            max_name_len = 255
        
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            for info in zip_ref.infolist():
                _extract_member(zip_ref, info, dest_dir, max_name_len)
        
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
    zip_path = p.DATASET_ROOT / "rwf2000.zip"
    
    if not download_file(url, zip_path, "RWF-2000 Dataset"):
        return False
    
    return extract_zip(zip_path, p.DATASET_ROOT, delete_after=True)


def download_affectnet() -> bool:
    """
    Baixa e extrai o dataset balanced-affectnet.

    Layout do zip (na raiz, sem 'archive (3)' e sem movimentação Train/Test):
        balanced-affectnet/
        ├── train/<Classe>/*.png
        ├── val/<Classe>/*.png
        └── test/<Classe>/*.png

    Sem labels.csv — as pastas de classe por split são a fonte da verdade
    para a ordem das classes no treinamento (BALAFF-01/02).
    """
    print("\n" + "="*60)
    print("DOWNLOAD: BALANCED-AFFECTNET")
    print("="*60)
    
    url = "https://www.kaggle.com/api/v1/datasets/download/dollyprajapati182/balanced-affectnet"
    affectnet_dir = p.DATASET_ROOT / "balanced-affectnet"
    
    # Idempotente: se a pasta alvo existe, não re-baixar (mesma convenção do download_rwf2000)
    if affectnet_dir.exists():
        print(f"[SKIP] Diretório já existe: {affectnet_dir}")
        return True
    
    zip_path = p.DATASET_ROOT / "balanced-affectnet.zip"
    
    if not download_file(url, zip_path, "Balanced-AffectNet Dataset"):
        return False
    
    if not extract_zip(zip_path, affectnet_dir, delete_after=True):
        return False
    
    # Enumerar e logar os nomes reais das pastas de classe por split
    # (fonte da verdade para a ordem das classes no treinamento).
    print("\n[INFO] Classes por split (ordenadas):")
    for split in ["train", "val", "test"]:
        split_dir = affectnet_dir / split
        if not split_dir.exists():
            print(f"[AVISO] Split não encontrado: {split_dir}")
            continue
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        print(f"  {split}: {len(classes)} classes -> {', '.join(classes)}")
    
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

  # Baixar AffectNet
  python download_datasets.py --affectnet
        """
    )
    
    # Modos de execução
    parser.add_argument("--all", action="store_true",
                       help="Baixar todos os datasets")
    parser.add_argument("--rwf2000", action="store_true",
                       help="Baixar dataset RWF-2000")
    parser.add_argument("--affectnet", action="store_true",
                       help="Baixar dataset AffectNet")
    
    args = parser.parse_args()
    
    # Verificar se pelo menos uma opção foi selecionada
    if not any([args.all, args.rwf2000, args.affectnet]):
        parser.print_help()
        print("\n[ERRO] Selecione pelo menos uma opção: --all, --rwf2000, --affectnet")
        return 1
    
    print("="*60)
    print("DOWNLOAD E PREPARAÇÃO DE DATASETS")
    print("="*60)
    print(f"\nDiretório de datasets: {p.DATASET_ROOT}")
    
    success = True

    # Cria diretório /dataset
    p.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    
    # --all: Baixar tudo
    if args.all:
        if not download_rwf2000():
            success = False
        if not download_affectnet():
            success = False
    
    # Downloads individuais
    if args.rwf2000:
        if not download_rwf2000():
            success = False
    
    if args.affectnet:
        if not download_affectnet():
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
