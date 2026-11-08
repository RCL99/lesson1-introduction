"""
@File: homework_tensors.py
@Description: homework tensors
@Date: 2026-11-07
@Author: RCL99
"""
import numpy as np
import torch


def create_tensors():
    """Задание 1.1: создание тензоров"""
    rand_3x4 = torch.tensor(np.random.rand(3, 4))
    zeros_2x3x4 = torch.tensor(np.zeros((2, 3, 4)))
    ones_5x5 = torch.tensor(np.ones((5, 5)))
    range_4x4 = torch.tensor(np.arange(16).reshape(4, 4))
    return {
        "rand_3x4": rand_3x4,
        "zeros_2x3x4": zeros_2x3x4,
        "ones_5x5": ones_5x5,
        "range_4x4": range_4x4,
    }


def tensor_operations():
    """Задание 1.2: операции с тензорами"""
    a = torch.tensor(np.random.rand(3, 4))
    b = torch.tensor(np.random.rand(4, 3))

    a_transposed = a.T
    matmul_ab = torch.matmul(a, b)
    elementwise = a * b.T
    total_sum = a.sum()

    assert a_transposed.shape == (4, 3)
    assert matmul_ab.shape == (3, 3)
    assert elementwise.shape == (3, 4)

    return {
        "A": a,
        "B": b,
        "A.T": a_transposed,
        "A @ B": matmul_ab,
        "A * B.T": elementwise,
        "sum(A)": total_sum,
    }


def indexing_and_slicing():
    """Задание 1.3: индексы и срезы"""
    tensor = torch.tensor(np.arange(125).reshape(5, 5, 5))

    first_row = tensor[0, 0, :]
    last_column = tensor[:, :, -1]
    center_2x2 = tensor[2, 2:4, 2:4]
    even_indices = tensor[::2, ::2, ::2]

    assert first_row.shape == (5,)
    assert last_column.shape == (5, 5)
    assert center_2x2.shape == (2, 2)
    assert even_indices.shape == (3, 3, 3)

    return {
        "first_row": first_row,
        "last_column": last_column,
        "center_2x2": center_2x2,
        "even_indices": even_indices,
    }


def reshape_tensor():
    """Задание 1.4: работа с формами"""
    tensor = torch.tensor(np.arange(24))
    shapes = (
        (2, 12),
        (3, 8),
        (4, 6),
        (2, 3, 4),
        (2, 2, 2, 3),
    )

    reshaped = {}
    for shape in shapes:
        result = tensor.reshape(shape)
        assert result.shape == shape
        reshaped[str(shape)] = result
    return reshaped


def print_tensor(name, tensor):
    """Вывод"""
    print(f"{name}: shape={tuple(tensor.shape)}")
    print(tensor)


def main():
    print("---1.1---")
    for name, tensor in create_tensors().items():
        print_tensor(name, tensor)
    print('\n')

    print("---1.2---")
    for name, tensor in tensor_operations().items():
        print_tensor(name, tensor)
    print('\n')

    print("---1.3---")
    for name, tensor in indexing_and_slicing().items():
        print_tensor(name, tensor)
    print('\n')

    print("---1.4---")
    for name, tensor in reshape_tensor().items():
        print_tensor(name, tensor)
    print('\n')


if __name__ == "__main__":
    main()
