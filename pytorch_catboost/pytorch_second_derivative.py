from typing import Callable, Tuple, Sequence, List

import numpy as np
import torch
from torch import Tensor


class LoglossObjective:
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = targets[index] - p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


class LoglossPytorchExplicit:
    def calc_ders_range(self, approxes, targets, weights):
        approxes = torch.FloatTensor(approxes)
        targets = torch.FloatTensor(targets)

        e = np.exp(approxes)
        p = e / (1 + e)
        der1 = targets - p
        der2 = -p * (1 - p)

        result = list(zip(der1.detach().numpy(), der2.detach().numpy()))
        return result


class LoglossPytorchAutograd:
    def calc_ders_range(self, approxes, targets, weights):
        approxes = torch.FloatTensor(approxes).requires_grad_()
        targets = torch.FloatTensor(targets)

        e = torch.exp(approxes)
        p = e / (1 + e)

        loss1 = targets * torch.log(p)
        loss0 = (1 - targets) * torch.log(1 - p)
        loss = (loss0 + loss1)
        loss = loss.sum()

        der1, = torch.autograd.grad(loss, approxes, create_graph=True)

        der2 = []
        for i in range(len(approxes)):
            _der2, = torch.autograd.grad(der1[i], approxes, create_graph=True)
            _der2_i = _der2[i].item()
            der2.append(_der2_i)

        der1 = der1.detach().numpy().copy()
        result = list(zip(der1, der2))
        return result


class CustomPytorchObjective:
    def __init__(self, loss_function: Callable[[Tensor, Tensor], Tensor]):
        self.loss_function = loss_function

    def calc_ders_range(self,
                        approxes: Sequence[float],
                        targets: Sequence[float],
                        weights: Sequence[float] = None
                        ) -> List[Tuple[float, float]]:
        approxes = torch.FloatTensor(approxes).requires_grad_()
        targets = torch.FloatTensor(targets)

        loss = self.loss_function(approxes, targets)
        der1, der2 = self._calculate_derivatives(loss, approxes)

        if weights is not None:
            weights = np.asarray(weights)
            der1, der2 = weights * der1, weights * der2

        result = list(zip(der1, der2))
        return result

    @staticmethod
    def _calculate_derivatives(loss: Tensor, approxes: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        der1, = torch.autograd.grad(loss, approxes, create_graph=True)

        der2 = []
        for i in range(len(approxes)):
            _der2, = torch.autograd.grad(der1[i], approxes, create_graph=True)
            _der2_i = _der2[i].item()
            der2.append(_der2_i)

        der1 = der1.detach().numpy()
        der2 = np.array(der2)
        return der1, der2


def logloss(approxes: Tensor, targets: Tensor) -> Tensor:
    e = torch.exp(approxes)
    p = e / (1 + e)

    loss1 = targets * torch.log(p)
    loss0 = (1 - targets) * torch.log(1 - p)
    loss = (loss0 + loss1)
    loss = loss.sum()

    return loss


def compare_methods():
    # TODO: check minus in loss func. Why doesn't the Patbas do it?
    n = 100
    approxes = np.random.randn(n)
    targets = np.random.randint(low=0, high=2, size=n)
    weights = None

    result_patbas = LoglossObjective().calc_ders_range(approxes, targets, weights)
    result_pytorch_explicit = LoglossPytorchExplicit().calc_ders_range(approxes, targets, weights)
    result_pytorch_autograd = LoglossPytorchAutograd().calc_ders_range(approxes, targets, weights)
    result_custom_pytorch = CustomPytorchObjective(loss_function=logloss).calc_ders_range(approxes, targets, weights)

    print(result_patbas)
    print(result_pytorch_explicit)
    print(result_pytorch_autograd)
    print(result_custom_pytorch)

    print(np.allclose(result_patbas, result_pytorch_explicit))
    print(np.allclose(result_patbas, result_pytorch_autograd, rtol=1e-4))
    print(np.allclose(result_patbas, result_custom_pytorch, rtol=1e-4))

    breakhere = 1


if __name__ == '__main__':
    compare_methods()
