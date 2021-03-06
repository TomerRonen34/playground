import numpy as np
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss

from pytorch_catboost.custom_pytorch_objective import CustomPytorchObjective
from pytorch_catboost.patbas_objectives import PatbasLoglossObjective, PatbasMseObjective


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


def logloss(approxes: Tensor, targets: Tensor) -> Tensor:
    e = torch.exp(approxes)
    p = e / (1 + e)

    loss1 = targets * torch.log(p)
    loss0 = (1 - targets) * torch.log(1 - p)
    loss = (loss0 + loss1)
    loss = loss.sum()

    return loss


def minus_bce_loss(approxes, targets):
    return - BCEWithLogitsLoss(reduction="sum")(approxes, targets)


def compare_methods_logloss():
    # TODO: check minus in loss func. Why doesn't the Patbas do it?
    n = 100
    approxes = np.random.randn(n)
    targets = np.random.randint(low=0, high=2, size=n)
    # weights = np.random.rand(n)
    weights = None

    result_patbas = PatbasLoglossObjective().calc_ders_range(approxes, targets, weights)
    result_pytorch_explicit = LoglossPytorchExplicit().calc_ders_range(approxes, targets, weights)
    result_pytorch_autograd = LoglossPytorchAutograd().calc_ders_range(approxes, targets, weights)
    result_custom_pytorch = CustomPytorchObjective(loss_function=logloss).calc_ders_range(approxes, targets, weights)
    result_custom_pytorch_builtin = CustomPytorchObjective(loss_function=BCEWithLogitsLoss(reduction="sum")
                                                           ).calc_ders_range(approxes, targets, weights)
    result_custom_pytorch_minus_builtin = CustomPytorchObjective(loss_function=minus_bce_loss
                                                                 ).calc_ders_range(approxes, targets, weights)

    print(result_patbas)
    print(result_pytorch_explicit)
    print(result_pytorch_autograd)
    print(result_custom_pytorch)
    print(result_custom_pytorch_builtin)
    print(result_custom_pytorch_minus_builtin)

    print(np.allclose(result_patbas, result_pytorch_explicit))
    print(np.allclose(result_patbas, result_pytorch_autograd, rtol=1e-4))
    print(np.allclose(result_patbas, result_custom_pytorch, rtol=1e-4))
    print(np.allclose(result_patbas, result_custom_pytorch_builtin, rtol=1e-4))
    print(np.allclose(result_patbas, result_custom_pytorch_minus_builtin, rtol=1e-4))

    breakhere = 1


def rmse_loss(approxes, targets):
    # return torch.sqrt(MSELoss(reduction="sum")(approxes, targets))
    return 0.5 * MSELoss(reduction="sum")(approxes, targets)


def basic_dml_loss(approxes: Tensor, targets: Tensor) -> Tensor:
    minimize_or_maximize = (-1) ** (1 - targets)
    # loss = minimize_or_maximize * ((approxes + 1e-4) ** 2)
    # loss = minimize_or_maximize * ((approxes - 1) ** 2)
    loss = minimize_or_maximize * torch.abs(approxes - 1)
    loss = loss.sum()
    return loss


def margin_dml_loss(approxes: Tensor, targets: Tensor, margin: float = 100) -> Tensor:
    loss = torch.abs(approxes - 1)
    is_different = targets == 0
    is_smaller_than_margin = approxes[is_different] ** 2 < margin
    loss[is_different] *= is_smaller_than_margin * -1
    loss = loss.sum()
    return loss


def compare_methods_rmse():
    n = 100
    approxes = np.random.randn(n)
    targets = np.random.randint(low=0, high=2, size=n)
    weights = None

    result_patbas = PatbasMseObjective().calc_ders_range(approxes, targets, weights)
    result_custom_pytorch2 = CustomPytorchObjective(loss_function=rmse_loss
                                                    ).calc_ders_range(approxes, targets, weights)

    print(result_patbas)
    print(result_custom_pytorch2)

    print(np.allclose(result_patbas, result_custom_pytorch2, rtol=1e-4))

    breakhere = 1


if __name__ == '__main__':
    compare_methods_logloss()
