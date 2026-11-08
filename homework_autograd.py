"""
@File: homework_autograd.py.py
@Description: homework autograd
@Date: 2026-11-08
@Author: RCL99
"""
import torch


def compute_simple_gradients():
    """Задание 2.1: простые вычисления с градиентами"""
    x = torch.tensor(6.0, requires_grad=True)
    y = torch.tensor(7.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
    f.backward()  # обратное распространение ошибки для вычисления градиента

    # df/dx = 2x + 2yz
    # df/dy = 2y + 2xz
    # df/dz = 2z + 2xy
    expected_x = 2 * x.detach() + 2 * y.detach() * z.detach()
    expected_y = 2 * y.detach() + 2 * x.detach() * z.detach()
    expected_z = 2 * z.detach() + 2 * x.detach() * y.detach()

    assert torch.allclose(x.grad, expected_x)
    assert torch.allclose(y.grad, expected_y)
    assert torch.allclose(z.grad, expected_z)

    return f.item(), x.grad.item(), y.grad.item(), z.grad.item()


def compute_mse_gradients():
    """Задание 2.2: градиенты MSE для y_pred = w * x + b"""
    x = torch.tensor([3.0, 6.0, 5.0, 8.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()

    return loss.item(), w.grad.item(), b.grad.item()


def compute_chain_rule_gradient():
    """Задание 2.3: цепное правило"""
    x = torch.tensor(3.0, requires_grad=True)

    f = torch.sin(x ** 2 + 1)

    # df/dx = cos(x^2+1) * 2x
    grad_formula = torch.cos(x ** 2 + 1) * 2 * x

    grad_autograd = torch.autograd.grad(f, x)[0]

    assert torch.allclose(grad_formula, grad_autograd)

    return f.item(), grad_formula.item(), grad_autograd.item()


def main():
    f_result, x_grad, y_grad, z_grad = compute_simple_gradients()
    print("---2.1---")
    print(f"Значение функции: {f_result}\nГрадиент x: {x_grad}\nГрадиент y: {y_grad}\nГрадиент z: {z_grad}\n")

    loss, w_grad, b_grad = compute_mse_gradients()
    print("---2.2---")
    print(f"MSE: {loss}\nГрадиент w: {w_grad}\nГрадиент b: {b_grad}\n")

    f_result, grad_formula, grad_autograd = compute_chain_rule_gradient()
    print("--2.3--")
    print(
        f"Значение функции: {f_result}\nГрадиент по формуле: {grad_formula}\nГрадиент через autograd: {grad_autograd}\n")


if __name__ == "__main__":
    main()
