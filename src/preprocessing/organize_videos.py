"""
Módulo para organizar vídeos do dataset RWF-2000 nas pastas corretas.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple

from src import paths as p


def _list_videos(directory: Path) -> list:
    """Lista arquivos .avi em um diretório, com fallback para os.listdir."""
    try:
        return list(directory.glob("*.avi"))
    except Exception:
        try:
            dir_str = str(directory.resolve())
            return [
                directory / fname
                for fname in os.listdir(dir_str)
                if fname.lower().endswith('.avi')
            ]
        except Exception as e:
            print(f"  Erro ao listar arquivos: {str(e)[:100]}")
            return []


def _get_safe_name(video_file: Path, fallback_index: int) -> str:
    try:
        return video_file.name
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        return f"video_{fallback_index:06d}.avi"


def _resolve_str(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def _safe_copy(src_file: Path, dest_file: Path, counter: int) -> bool:
    """Copia arquivo de forma segura. Tenta 3 estratégias: nome original, seguro, absoluto."""
    try:
        src_str = _resolve_str(src_file)
        if not os.path.exists(src_str):
            return False
        if not dest_file.exists():
            dest_str = _resolve_str(dest_file)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_str, dest_str)
        return True
    except (OSError, UnicodeEncodeError, UnicodeDecodeError, PermissionError):
        try:
            safe_name = f"video_{counter:06d}.avi"
            dest_safe = dest_file.parent / safe_name
            if not dest_safe.exists():
                src_str = _resolve_str(src_file)
                if os.path.exists(src_str):
                    shutil.copy2(src_str, _resolve_str(dest_safe))
                    print(f"  Arquivo com encoding problemático renomeado: {safe_name}")
            return True
        except Exception:
            try:
                src_abs = os.path.abspath(str(src_file))
                if os.path.exists(src_abs):
                    safe_name = f"video_{counter:06d}.avi"
                    dest_abs = os.path.join(str(dest_file.parent.resolve()), safe_name)
                    if not os.path.exists(dest_abs):
                        shutil.copy2(src_abs, dest_abs)
                        return True
            except:
                pass
            return False


def _process_class_videos(split_path: Path, class_name: str, dest_dir: Path,
                          counter_ref: list, errors_ref: list):
    """Processa vídeos de uma classe (Fight/NonFight)."""
    class_dir = split_path / class_name
    if not class_dir.exists():
        return

    label = "violento" if class_name == "Fight" else "não violento"
    print(f"Processando vídeos {class_name} ({label}) de {split_path.name}...")

    video_files = _list_videos(class_dir)
    print(f"  Encontrados {len(video_files)} vídeos")

    for idx, video_file in enumerate(video_files, 1):
        try:
            video_name = _get_safe_name(video_file, counter_ref[0] + idx)
            if isinstance(video_file, str):
                video_file = Path(video_file)
            dest_path = dest_dir / video_name
            if _safe_copy(video_file, dest_path, counter_ref[0] + idx):
                counter_ref[0] += 1
            else:
                errors_ref[0] += 1
            if idx % 100 == 0:
                print(f"  Processados {idx}/{len(video_files)} vídeos...")
        except Exception as e:
            print(f"  Erro ao processar arquivo (índice {idx}): {str(e)[:100]}")
            errors_ref[0] += 1


def organize_rwf2000_dataset(
    dataset_root: str = None,
    output_root: str = None
) -> Tuple[int, int]:
    if dataset_root is None:
        dataset_root = str(p.RWF2000_ROOT)
    if output_root is None:
        output_root = str(p.RAW_DATA_ROOT)
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)

    num_violent, num_non_violent, num_errors = [0], [0], [0]

    for split in ["train", "val"]:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Aviso: Pasta {split_path} não encontrada. Pulando...")
            continue

        violent_dir = output_path / split / "violent"
        non_violent_dir = output_path / split / "non_violent"
        violent_dir.mkdir(parents=True, exist_ok=True)
        non_violent_dir.mkdir(parents=True, exist_ok=True)

        _process_class_videos(split_path, "Fight", violent_dir, num_violent, num_errors)
        _process_class_videos(split_path, "NonFight", non_violent_dir, num_non_violent, num_errors)

    vt, nvt, err = num_violent[0], num_non_violent[0], num_errors[0]
    print(f"\nOrganização concluída:")
    print(f"  - Vídeos violentos: {vt}")
    print(f"  - Vídeos não violentos: {nvt}")
    print(f"  - Total: {vt + nvt}")
    if err > 0:
        print(f"  - Erros encontrados: {err}")
    return vt, nvt


if __name__ == "__main__":
    # Executar organização
    organize_rwf2000_dataset()

