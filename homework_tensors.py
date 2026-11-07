"""
@File: homework_tensors.py
@Description: Homework 1. Tensor creation.
@Date: 2026-11-07
@Author: RCL99
"""
import torch


def create_tensors():
    """Задание 1.
    Создайте следующие тензоры:
    - Тензор размером 3x4, заполненный случайными числами от 0 до 1
    - Тензор размером 2x3x4, заполненный нулями
    - Тензор размером 5x5, заполненный единицами
    - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
    """
    rand_3x4 = torch.rand(3, 4)
    zeros_2x3x4 = torch.zeros(2, 3, 4)
    ones_5x5 = torch.ones(5, 5)
    range_4x4 = torch.arange(16).reshape(4, 4)
    return rand_3x4, zeros_2x3x4, ones_5x5, range_4x4


def main():
    # names = ("rand_3x4", "zeros_2x3x4", "ones_5x5", "range_4x4")
    # for name, tensor in zip(names, create_tensors()):
    #     print(name, tensor.shape, tensor.dtype)
    #     print(tensor)
    create_tensors()

if __name__ == "__main__":
    main()
