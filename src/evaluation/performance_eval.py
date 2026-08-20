"""
Avaliação de performance em tempo real.

Mede:
- FPS (Frames Per Second)
- Latência (tempo por janela)
- Uso de recursos (CPU, GPU, Memória)
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
import psutil
import platform
from collections import deque

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from .utils import save_results, create_experiment_dir


class PerformanceEvaluator:
    """
    Avaliador de performance em tempo real.
    """
    
    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        warmup_iterations: int = 10
    ):
        """
        Inicializa o avaliador de performance.
        
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
        
        # Inicializar NVML se disponível
        if HAS_NVML and torch.cuda.is_available():
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None
    
    def measure_fps(
        self,
        num_iterations: int = 100,
        include_capture: bool = False
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
        
        # Warmup
        print("Warming up...")
        for _ in range(self.warmup_iterations):
            batch = next(iter(self.dataloader))
            self._process_batch(batch)
        
        # Medição
        print(f"Measuring FPS ({num_iterations} iterations)...")
        start_time = time.time()
        
        for i in range(num_iterations):
            iter_start = time.time()
            
            if include_capture:
                # Simular captura
                capture_start = time.time()
                batch = next(iter(self.dataloader))
                capture_time = time.time() - capture_start
            else:
                batch = next(iter(self.dataloader))
                capture_time = 0
            
            # Processar
            process_time = self._process_batch(batch)
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            if iter_time > 0:
                fps = 1.0 / iter_time
                fps_list.append(fps)
        
        total_time = time.time() - start_time
        avg_fps = num_iterations / total_time
        
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
                "p99": float(np.percentile(fps_list, 99))
            },
            "avg_fps": float(avg_fps),
            "total_time": float(total_time),
            "num_iterations": num_iterations,
            "iteration_times": {
                "mean": float(np.mean(iteration_times)),
                "std": float(np.std(iteration_times)),
                "min": float(np.min(iteration_times)),
                "max": float(np.max(iteration_times))
            },
            "capture_time": float(capture_time) if include_capture else None
        }
    
    def measure_latency(
        self,
        num_iterations: int = 100
    ) -> Dict:
        """
        Mede latência por componente.
        
        Args:
            num_iterations: Número de iterações
        
        Returns:
            Dicionário com latências por componente
        """
        latencies = {
            "capture": [],
            "preprocessing": [],
            "inference": [],
            "postprocessing": [],
            "total": []
        }
        
        # Warmup
        for _ in range(self.warmup_iterations):
            batch = next(iter(self.dataloader))
            self._process_batch(batch)
        
        print(f"Measuring latency ({num_iterations} iterations)...")
        
        for _ in range(num_iterations):
            # Captura
            capture_start = time.time()
            batch = next(iter(self.dataloader))
            capture_time = time.time() - capture_start
            latencies["capture"].append(capture_time)
            
            # Preprocessing
            prep_start = time.time()
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs = batch[0:-1]
            else:
                inputs = batch
            
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, (list, tuple)):
                inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
            prep_time = time.time() - prep_start
            latencies["preprocessing"].append(prep_time)
            
            # Inference
            inf_start = time.time()
            with torch.no_grad():
                if isinstance(inputs, torch.Tensor):
                    outputs = self.model(inputs)
                elif isinstance(inputs, (list, tuple)):
                    outputs = self.model(*inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inf_time = time.time() - inf_start
            latencies["inference"].append(inf_time)
            
            # Postprocessing
            post_start = time.time()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            post_time = time.time() - post_start
            latencies["postprocessing"].append(post_time)
            
            # Total
            total_time = capture_time + prep_time + inf_time + post_time
            latencies["total"].append(total_time)
        
        # Calcular estatísticas
        results = {}
        for component, times in latencies.items():
            results[component] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "median": float(np.median(times)),
                "percentiles": {
                    "p50": float(np.percentile(times, 50)),
                    "p90": float(np.percentile(times, 90)),
                    "p95": float(np.percentile(times, 95)),
                    "p99": float(np.percentile(times, 99))
                }
            }
        
        return results
    
    def measure_resource_usage(
        self,
        duration: float = 10.0,
        sample_interval: float = 0.1
    ) -> Dict:
        """
        Mede uso de recursos durante execução.
        
        Args:
            duration: Duração da medição em segundos
            sample_interval: Intervalo entre amostras
        
        Returns:
            Dicionário com estatísticas de uso de recursos
        """
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        gpu_memory = []
        gpu_temperature = []
        
        process = psutil.Process()
        num_samples = int(duration / sample_interval)
        
        print(f"Measuring resource usage ({duration}s, {num_samples} samples)...")
        
        # Warmup
        for _ in range(self.warmup_iterations):
            batch = next(iter(self.dataloader))
            self._process_batch(batch)
        
        # Medição
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            # CPU e Memória
            cpu_percent = process.cpu_percent(interval=None)
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024 ** 3)
            
            cpu_usage.append(cpu_percent)
            memory_usage.append(memory_gb)
            
            # GPU (se disponível)
            if self.handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_usage.append(util.gpu)
                    gpu_memory.append(mem_info.used / (1024 ** 3))  # GB
                    gpu_temperature.append(temp)
                except:
                    pass
            
            # Processar batch
            batch = next(iter(self.dataloader))
            self._process_batch(batch)
            
            iteration += 1
            time.sleep(sample_interval)
        
        # Calcular estatísticas
        results = {
            "cpu": {
                "mean": float(np.mean(cpu_usage)),
                "std": float(np.std(cpu_usage)),
                "max": float(np.max(cpu_usage))
            },
            "memory": {
                "mean": float(np.mean(memory_usage)),
                "std": float(np.std(memory_usage)),
                "max": float(np.max(memory_usage))
            }
        }
        
        if len(gpu_usage) > 0:
            results["gpu"] = {
                "usage": {
                    "mean": float(np.mean(gpu_usage)),
                    "std": float(np.std(gpu_usage)),
                    "max": float(np.max(gpu_usage))
                },
                "memory": {
                    "mean": float(np.mean(gpu_memory)),
                    "std": float(np.std(gpu_memory)),
                    "max": float(np.max(gpu_memory))
                },
                "temperature": {
                    "mean": float(np.mean(gpu_temperature)),
                    "std": float(np.std(gpu_temperature)),
                    "max": float(np.max(gpu_temperature))
                }
            }
        
        # Informações do sistema
        results["system"] = {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "device": str(self.device)
        }
        
        return results
    
    def _process_batch(self, batch):
        """Processa um batch e retorna tempo."""
        start = time.time()
        
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs = batch[0]
        else:
            inputs = batch
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, (list, tuple)):
            inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                outputs = self.model(inputs)
            elif isinstance(inputs, (list, tuple)):
                outputs = self.model(*inputs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        return time.time() - start
    
    def evaluate_all(
        self,
        output_dir: str,
        experiment_name: str = "performance",
        num_fps_iterations: int = 100,
        num_latency_iterations: int = 100,
        resource_duration: float = 10.0
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
        
        # FPS
        print("\n" + "="*60)
        print("Measuring FPS...")
        print("="*60)
        fps_results = self.measure_fps(num_fps_iterations)
        results["fps"] = fps_results
        save_results(fps_results, output_path / "fps_report.json")
        
        # Latência
        print("\n" + "="*60)
        print("Measuring Latency...")
        print("="*60)
        latency_results = self.measure_latency(num_latency_iterations)
        results["latency"] = latency_results
        save_results(latency_results, output_path / "latency_report.json")
        
        # Recursos
        print("\n" + "="*60)
        print("Measuring Resource Usage...")
        print("="*60)
        resource_results = self.measure_resource_usage(resource_duration)
        results["resources"] = resource_results
        save_results(resource_results, output_path / "resource_usage.json")
        
        # Salvar resumo
        save_results(results, output_path / "performance_summary.json")
        
        return results

