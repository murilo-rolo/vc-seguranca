from src import paths as p

EMOTION_MODEL_PATH = p.EMOTION_CNN_WEIGHTS / "best_model.pth"


def _check_dataset_root() -> bool:
    if not p.DATASET_ROOT.exists():
        print(f"Erro: Diretório de datasets não encontrado: {p.DATASET_ROOT}")
        print("  Certifique-se de que os datasets estão em 'dataset/RWF-2000'")
        return False
    return True


def _print_header(title: str, fields: dict):
    print("=" * 60)
    print(title)
    print("=" * 60)
    for key, value in fields.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print()


def _process_rwf2000_pose(num_frames, model_complexity, conf=0.5, iou=0.7):
    from src.pose.extract_pose import process_dataset_for_pose

    rwf2000_path = p.DATASET_ROOT / "RWF-2000"
    if rwf2000_path.exists():
        print("\n" + "=" * 60)
        print("Processando RWF-2000...")
        print("=" * 60)
        process_dataset_for_pose(
            dataset_root=str(rwf2000_path),
            output_root=str(p.POSE_ROOT),
            dataset_name="rwf2000",
            num_frames=num_frames,
            model_complexity=model_complexity,
            conf=conf,
            iou=iou,
        )
    else:
        print(f"\nAviso: Dataset RWF-2000 não encontrado em {rwf2000_path}")
        print("  Pulando processamento de RWF-2000...")
