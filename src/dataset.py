import torch


def get_linear_data(n: int = 1000) -> torch.tensor:
    X = torch.normal(0.0, 1, (n, 2))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b
    return data