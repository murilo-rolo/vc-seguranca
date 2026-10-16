"""
Avaliação de performance em tempo real.

Mede:
- FPS (Frames Per Second)
- Latência (tempo por janela)
- Uso de recursos (CPU, GPU, Memória)
"""

import platform
import time
from typing import Dict, Iterator

import numpy as np
import psutil
import torch

from src.tensor_utils import move_to_device

from .inference import forward_batch, unpack_batch
from .utils import create_experiment_dir, save_results

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class PerformanceEvaluator:
    """
    Avaliador de performance em tempo real.
    """

    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        warmup_iterations: int = 10,
    ):
        """
        Args:
            model: Modelo PyTorch
            dataloader: DataLoader com dados de teste
            device: Device para inferência
            warmup_iterations: Número de iterações de warmup
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.warmup_iterations = warmup_iterations
        self.model.eval()

        if HAS_NVML and torch.cuda.is_available():
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def _batches(self) -> Iterator:
        """Stream infinito de batches (repassa o dataloader quando termina)."""
        while True:
            for batch in self.dataloader:
                yield batch

    def _warmup(self, stream: Iterator) -> None:
        for _ in range(self.warmup_iterations):
            self._process_batch(next(stream))

    def _process_batch(self, batch) -> float:
        """Processa um batch e retorna o tempo gasto."""
        start = time.time()
        inputs, _ = unpack_batch(batch)
        inputs = move_to_device(inputs, self.device)
        with torch.no_grad():
            forward_batch(self.model, inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - start

    def measure_fps(
        self,
        num_iterations: int = 100,
        include_capture: bool = False,
    ) -> Dict:
        """
        Mede FPS do modelo.

        Args:
            num_iterations: Número de iterações para medir
            include_capture: Se True, inclui tempo de captura

        Returns:
            Dicionário com estatísticas de FPS
        """
        fps_list = []
        iteration_times = []
        capture_time = 0.0

        stream = self._batches()
        print("Warming up...")
        self._warmup(stream)

        print(f"Measuring FPS ({num_iterations} iterations)...")
        start_time = time.time()

        for _ in range(num_iterations):
            iter_start = time.time()

            capture_start = time.time()
            batch = next(stream)
            capture_time = time.time() - capture_start if include_capture else 0.0

            self._process_batch(batch)

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            if iter_time > 0:
                fps_list.append(1.0 / iter_time)

        total_time = time.time() - start_time

        return {
            "fps_mean": float(np.mean(fps_list)),
            "fps_std": float(np.std(fps_list)),
            "fps_min": float(np.min(fps_list)),
            "fps_max": float(np.max(fps_list)),
            "fps_median": float(np.median(fps_list)),
            "fps_percentiles": {
                "p50": float(np.percentile(fps_list, 50)),
                "p90": float(np.percentile(fps_list, 90)),
                "p95": float(np.percentile(fps_list, 95)),
                "p99": float(np.percentile(fps_list, 99)),
            },
            "avg_fps": float(num_iterations / total_time),
            "total_time": float(total_time),
            "num_iterations": num_iterations,
            "iteration_times": {
                "mean": float(np.mean(iteration_times)),
                "std": float(np.std(iteration_times)),
                "min": float(np.min(iteration_times)),
                "max": float(np.max(iteration_times)),
            },
            "capture_time": float(capture_time) if include_capture else None,
        }

    def measure_latency(self, num_iterations: int = 100) -> Dict:
        """
        Mede latência por componente (captura, pré-processamento, inferência,
        pós-processamento).

        Args:
            num_iterations: Número de iterações

        Returns:
            Dicionário com latências por componente
        """
        latencies = {k: [] for k in ("capture", "preprocessing", "inference", "postprocessing", "total")}

        stream = self._batches()
        self._warmup(stream)

        print(f"Measuring latency ({num_iterations} iterations)...")
        for _ in range(num_iterations):
            capture_start = time.time()
            batch = next(stream)
            capture_time = time.time() - capture_start

            prep_start = time.time()
            inputs, _ = unpack_batch(batch)
            inputs = move_to_device(inputs, self.device)
            prep_time = time.time() - prep_start

            inf_start = time.time()
            with torch.no_grad():
                outputs = forward_batch(self.model, inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inf_time = time.time() - inf_start

            post_start = time.time()
            torch.softmax(outputs, dim=1)
            torch.argmax(outputs, dim=1)
            post_time = time.time() - post_start

            latencies["capture"].append(capture_time)
            latencies["preprocessing"].append(prep_time)
            latencies["inference"].append(inf_time)
            latencies["postprocessing"].append(post_time)
            latencies["total"].append(capture_time + prep_time + inf_time + post_time)

        return {component: _stats(times) for component, times in latencies.items()}

    def measure_resource_usage(
        self,
        duration: float = 10.0,
        sample_interval: float = 0.1,
    ) -> Dict:
        """
        Mede uso de recursos durante execução.

        Args:
            duration: Duração da medição em segundos
            sample_interval: Intervalo entre amostras

        Returns:
            Dicionário com estatísticas de uso de recursos
        """
        cpu_usage, memory_usage = [], []
        gpu_usage, gpu_memory, gpu_temperature = [], [], []

        process = psutil.Process()
        stream = self._batches()
        self._warmup(stream)

        print(f"Measuring resource usage ({duration}s)...")
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_usage.append(process.cpu_percent(interval=None))
            memory_usage.append(process.memory_info().rss / (1024 ** 3))

            if self.handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_usage.append(util.gpu)
                    gpu_memory.append(mem_info.used / (1024 ** 3))
                    gpu_temperature.append(temp)
                except Exception:
                    pass

            self._process_batch(next(stream))
            time.sleep(sample_interval)

        results = {
            "cpu": _stats(cpu_usage, percentiles=False),
            "memory": _stats(memory_usage, percentiles=False),
            "system": {
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
                "device": str(self.device),
            },
        }

        if gpu_usage:
            results["gpu"] = {
                "usage": _stats(gpu_usage, percentiles=False),
                "memory": _stats(gpu_memory, percentiles=False),
                "temperature": _stats(gpu_temperature, percentiles=False),
            }

        return results

    def evaluate_all(
        self,
        output_dir: str,
        experiment_name: str = "performance",
        num_fps_iterations: int = 100,
        num_latency_iterations: int = 100,
        resource_duration: float = 10.0,
    ) -> Dict:
        """
        Executa todas as avaliações de performance.

        Args:
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
            num_fps_iterations: Iterações para FPS
            num_latency_iterations: Iterações para latência
            resource_duration: Duração para medição de recursos

        Returns:
            Dicionário com todos os resultados
        """
        output_path = create_experiment_dir(output_dir, experiment_name)
        results = {}

        print("\n" + "=" * 60)
        print("Measuring FPS...")
        print("=" * 60)
        results["fps"] = self.measure_fps(num_fps_iterations)
        save_results(results["fps"], output_path / "fps_report.json")

        print("\n" + "=" * 60)
        print("Measuring Latency...")
        print("=" * 60)
        results["latency"] = self.measure_latency(num_latency_iterations)
        save_results(results["latency"], output_path / "latency_report.json")

        print("\n" + "=" * 60)
        print("Measuring Resource Usage...")
        print("=" * 60)
        results["resources"] = self.measure_resource_usage(resource_duration)
        save_results(results["resources"], output_path / "resource_usage.json")

        save_results(results, output_path / "performance_summary.json")
        return results


def _stats(values, percentiles: bool = True) -> Dict:
    """Estatísticas descritivas de uma lista de amostras."""
    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    if percentiles:
        stats["median"] = float(np.median(values))
        stats["p50"] = float(np.percentile(values, 50))
        stats["p90"] = float(np.percentile(values, 90))
        stats["p95"] = float(np.percentile(values, 95))
        stats["p99"] = float(np.percentile(values, 99))
    return stats
