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
import concurrent.futures
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePath
from typing import List, Optional

_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
_CHUNK = 1024 * 1024  # 1 MB por leitura
_MIN_SPLIT = 16 * _CHUNK  # abaixo disso, não vale segmentar (aria2c usa --min-split-size=1M x16)

from src import paths as p


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """
    Faz download de um arquivo, preferindo aria2c (16 conexões paralelas).

    Se o aria2c não estiver instalado — ou falhar na execução — cai num
    fallback em Python puro que espelha o mesmo comportamento: downloads
    segmentados via Range requests em 16 conexões (ThreadPoolExecutor).

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

    if shutil.which("aria2c"):
        if _download_aria2c(url, dest, description):
            return True
        print("[AVISO] aria2c falhou; tentando fallback paralelo...")
    else:
        print("[AVISO] aria2c não encontrado; usando fallback paralelo (16 conexões).")
        print("  Dica: sudo apt install aria2 | brew install aria2 | choco install aria2")

    return _download_fallback(url, dest, description)


def _download_aria2c(url: str, dest: Path, description: str = "") -> bool:
    """Baixa via aria2c com 16 conexões paralelas. Retorna True se sucesso."""
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


def _download_fallback(url: str, dest: Path, description: str = "", connections: int = 16) -> bool:
    """
    Fallback em Python puro (stdlib) quando o aria2c não está disponível/falhou.

    Estratégia (mesmo princípio do aria2c -x16 -s16):
      1. Probe: GET com 'Range: bytes=0-0' — segue redirecionamentos e descobre a
         URL final, o tamanho total (via Content-Range) e se o servidor suporta Range.
      2. Se suporta (206): baixa em 'connections' segmentos paralelos, gravando cada
         um no offset exato de um arquivo pré-alocado (os.pwrite). Retoma por
         segmento em caso de falha.
      3. Se não suporta (200): stream único de 1MB com retries.
    """
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()

    print(f"[DOWNLOAD] {description or dest.name}")
    print(f"  URL: {url}")
    print(f"  Destino: {dest}")
    print("  Motor: fallback paralelo (Range requests, stdlib)")

    try:
        req = urllib.request.Request(
            url, method="GET",
            headers={"Range": "bytes=0-0", "User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            final_url = resp.geturl()
            total = 0
            content_range = resp.headers.get("Content-Range")
            if status == 206 and content_range:
                try:
                    total = int(content_range.rsplit("/", 1)[1])
                except ValueError:
                    total = 0
    except Exception as e:
        print(f"[ERRO] Falha ao conectar: {e}")
        return False

    if status == 206 and total > 0:
        return _download_parallel(final_url, part, dest, total, description, connections)

    return _download_single(final_url, part, dest, 0, description)


def _download_single(url: str, part: Path, dest: Path, total: int, description: str) -> bool:
    """Stream único (sem Range) com retries e verificação de tamanho."""
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if total == 0:
                    try:
                        total = int(resp.headers.get("Content-Length") or 0)
                    except ValueError:
                        total = 0
                with open(part, "wb") as f:
                    while True:
                        data = resp.read(_CHUNK)
                        if not data:
                            break
                        f.write(data)
                break
        except Exception as e:
            if attempt < 4:
                print(f"  [RETRY] {e}; tentativa {attempt + 2}/5", flush=True)
                time.sleep(3)
                continue
            if part.exists():
                part.unlink()
            print(f"[ERRO] Download falhou após 5 tentativas: {dest.name}")
            return False

    size = part.stat().st_size
    if total and size != total:
        print(f"[ERRO] Tamanho incorreto: esperado {total}, obtido {size}")
        part.unlink()
        return False
    os.replace(part, dest)
    print(f"[OK] Download concluído: {dest} ({size / (1024 * 1024):.1f} MB)")
    return True


def _download_parallel(url: str, part: Path, dest: Path, total: int, description: str, connections: int = 16) -> bool:
    """
    Download segmentado em paralelo via Range requests.

    Cada worker abre sua própria conexão para 'url' (já resolvida pelo probe),
    baixa 'Range: bytes=inicio-fim' e grava no offset exato de um arquivo
    pré-alocado via os.pwrite (seguro entre threads). Em falha, o segmento
    retoma de onde parou ('start+written').
    """
    if total < _MIN_SPLIT:
        print(f"  Arquivo pequeno ({total / 1024 / 1024:.1f} MB): usando stream único")
        return _download_single(url, part, dest, total, description)

    connections = max(1, min(connections, total // _CHUNK))
    seg_size = math.ceil(total / connections)
    actual_conn = (total + seg_size - 1) // seg_size

    print(f"  Segmentos: {actual_conn} ({connections} conexões), ~{seg_size / 1024 / 1024:.0f} MB cada")

    fd = None
    try:
        fd = os.open(part, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        os.ftruncate(fd, total)
    except OSError as e:
        if fd is not None:
            os.close(fd)
        if part.exists():
            part.unlink()
        print(f"[ERRO] Falha ao pré-alocar {part}: {e}")
        return False

    progress = {"done": 0}
    progress_lock = threading.Lock()
    last_mark = {"mb": 0}

    def report(amount: int) -> None:
        with progress_lock:
            progress["done"] += amount
            mark = progress["done"] // (512 * 1024 * 1024)
            if mark > last_mark["mb"]:
                last_mark["mb"] = mark
                pct = progress["done"] * 100.0 / total
                print(f"  {progress['done'] / 1024**3:.2f} GB / {total / 1024**3:.2f} GB ({pct:.1f}%)", flush=True)

    def worker(index: int) -> int:
        start = index * seg_size
        end = min(total, start + seg_size) - 1
        expected = end - start + 1
        written = 0
        for attempt in range(5):
            try:
                if written >= expected:
                    return written
                headers = {"User-Agent": _USER_AGENT, "Range": f"bytes={start + written}-{end}"}
                req = urllib.request.Request(url, method="GET", headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 206:
                        raise urllib.error.HTTPError(url, resp.status, "servidor não respondeu 206", resp.headers, None)
                    while written < expected:
                        data = resp.read(_CHUNK)
                        if not data:
                            break
                        os.pwrite(fd, data, start + written)
                        written += len(data)
                        report(len(data))
                if written < expected:
                    raise OSError(f"conexão encerrada cedo ({written}/{expected})")
                return written
            except Exception as e:
                if attempt < 4:
                    print(f"  [RETRY] segmento {index + 1}/{actual_conn}: {e} (tentativa {attempt + 2}/5)", flush=True)
                    time.sleep(3)
                    continue
                raise
        return written

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as ex:
            futures = [ex.submit(worker, i) for i in range(actual_conn)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
        os.fsync(fd)
        os.close(fd)
        fd = None
        if part.stat().st_size != total:
            raise OSError(f"tamanho final incorreto: {part.stat().st_size} != {total}")
        os.replace(part, dest)
    except Exception as e:
        if fd is not None:
            os.close(fd)
        if part.exists():
            part.unlink()
        print(f"[ERRO] Download paralelo falhou: {e}")
        return False

    print(f"[OK] Download concluído: {dest} ({total / (1024 * 1024):.1f} MB)")
    return True


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

def organize_emotion_classes(emotion_dataset: Path) -> bool:
    """
    Organiza as classes Anger, Surprise e Fear sob violent/, e as classes Neutral, Happy e Sad sob non_violent/.
    Disgust e Contempt são excluídos por serem ambíguas e não se encaixarem bem no contexto da aplicação.
    As imagens são renomeadas com o prefixo da emoção + sequência numérica (ex: anger_00001.png).

    Estrutura resultante por split:
        balanced-affectnet/
        ├── train/
        │   ├── violent/anger_00001.png, anger_00002.png, ...
        │   └── non_violent/neutral_00001.png, happy_00001.png, ...
        ├── val/
        │   ├── violent/...
        │   └── non_violent/...
        └── test/
            ├── violent/...
            └── non_violent/...

    Args:
        emotion_dataset: Path do diretório que foi baixado o dataset de emoção

    Returns:
        True se sucesso, False se erro
    """
    if not emotion_dataset.exists():
        print(f"[ERRO] Diretório do dataset de emoção não encontrado: {emotion_dataset}")
        return False

    emotion_to_dir = {
        "Anger": "violent", "Surprise": "violent", "Fear": "violent",
        "Neutral": "non_violent", "Happy": "non_violent", "Sad": "non_violent",
    }

    for folder in ["test", "val", "train"]:
        folder2 = emotion_dataset / folder

        for emotion in ["Contempt", "Disgust"]:
            folder3 = folder2 / emotion
            if folder3.exists():
                shutil.rmtree(folder3)

        for emotion, target_subdir in emotion_to_dir.items():
            src_dir = folder2 / emotion
            target_dir = folder2 / target_subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                continue

            emotion_files = sorted(src_dir.glob("*.png"))
            if not emotion_files:
                emotion_files = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.jpeg"))

            for i, img_path in enumerate(emotion_files):
                suffix = img_path.suffix
                new_name = f"{emotion.lower()}_{i+1:05d}{suffix}"
                dst_path = target_dir / new_name
                if dst_path.exists():
                    shutil.rmtree(dst_path) if dst_path.is_dir() else dst_path.unlink()
                shutil.copy2(str(img_path), str(dst_path))

            shutil.rmtree(src_dir)

    return True


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
    
    # Idempotente: se o dataset já existe e está organizado, não re-baixar
    val_dir = affectnet_dir / "val"
    if affectnet_dir.exists() and (val_dir / "violent").exists():
        print(f"[SKIP] Dataset já organizado: {affectnet_dir}")
        return True
    
    if affectnet_dir.exists():
        shutil.rmtree(affectnet_dir)
    
    zip_path = p.DATASET_ROOT / "balanced-affectnet.zip"
    
    if not download_file(url, zip_path, "Balanced-AffectNet Dataset"):
        return False
    
    if not extract_zip(zip_path, affectnet_dir, delete_after=True):
        return False

    if not organize_emotion_classes(affectnet_dir):
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
