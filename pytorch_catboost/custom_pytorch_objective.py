from typing import Callable, Sequence, List, Tuple

import numpy as np
import torch
from torch import Tensor

LossFunctionType = Callable[[Tensor, Tensor], Tensor]


class CustomPytorchObjective:
    def __init__(self, loss_function: LossFunctionType):
        self.loss_function = loss_function

    def calc_ders_range(self,
                        approxes: Sequence[float],
                        targets: Sequence[float],
                        weights: Sequence[float] = None
                        ) -> List[Tuple[float, float]]:
        approxes = torch.FloatTensor(approxes).requires_grad_()
        targets = torch.FloatTensor(targets)

        objective = - self.loss_function(approxes, targets)
        der1, der2 = self._calculate_derivatives(objective, approxes)

        if weights is not None:
            weights = np.asarray(weights)
            der1, der2 = weights * der1, weights * der2

        result = list(zip(der1, der2))
        return result

    @staticmethod
    def _calculate_derivatives(objective: Tensor, approxes: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        der1, = torch.autograd.grad(objective, approxes, create_graph=True)

        der2 = []
        for i in range(len(approxes)):
            _der2, = torch.autograd.grad(der1[i], approxes, create_graph=True)
            _der2_i = _der2[i].item()
            der2.append(_der2_i)

        der1 = der1.detach().numpy()
        der2 = np.array(der2)
        return der1, der2
