"""
@File: homework_performance.py
@Description: homework performance
@Date: 2026-11-08
@Author: RCL99
"""
import time

import torch
from prettytable import PrettyTable


def create_matrices():
    """Задание 3.1: подготовка данных"""
    return {
        "64x1024x1024": torch.randn(64, 1024, 1024),
        "128x512x512": torch.randn(128, 512, 512),
        "256x256x256": torch.randn(256, 256, 256),
    }


def measure_time(operation, device):
    """Задание 3.2: время операции в миллисекундах"""
    if device.type == "cuda":
        operation()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        operation()
        end.record()

        torch.cuda.synchronize()
        return start.elapsed_time(end)

    start = time.time()
    operation()
    return (time.time() - start) * 1000


def time_comparison(matrices):
    """Задание 3.3: сравнение операций на CPU и CUDA"""
    operations = {
        "Матричное умножение": lambda tensor: torch.matmul(tensor, tensor),
        "Сложение": lambda tensor: tensor + tensor,
        "Умножение": lambda tensor: tensor * tensor,
        "Транспонирование": lambda tensor: tensor.transpose(-2, -1),
        "Сумма": lambda tensor: tensor.sum(),
    }

    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if torch.cuda.is_available() else None

    if gpu is None:
        print("CUDA нет")

    for name, tensor in matrices.items():
        gpu_tensor = tensor.to(gpu) if gpu is not None else None

        table = PrettyTable()
        table.field_names = ["Операция", "CPU (мс)", "GPU (мс)", "Ускорение"]

        for operation_name, operation in operations.items():
            cpu_ms = measure_time(lambda: operation(tensor), cpu)

            if gpu_tensor is None:
                table.add_row([operation_name, f"{cpu_ms:.3f}", "-", "-"])
                continue

            gpu_ms = measure_time(lambda: operation(gpu_tensor), gpu)
            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
            table.add_row(
                [operation_name, f"{cpu_ms:.3f}", f"{gpu_ms:.3f}", f"{speedup:.1f}x"]
            )

        print(f"\nМатрица {name}")
        print(table)


def main():
    matrices = create_matrices()
    time_comparison(matrices)


if __name__ == "__main__":
    main()
