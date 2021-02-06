import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
import torch

from pytorch_catboost.custom_pytorch_objective import CustomPytorchObjective


def plot_difference(pred, y, label="", title=None, is_show=True):
    diff = (pred - y) / y
    plt.hist(diff, bins=15, alpha=0.4, label=label)
    if title is not None:
        plt.title(title)
    if is_show:
        plt.legend()
        plt.show()


def custom_loss(approxes, targets):
    loss = torch.zeros_like(approxes)
    is_bigger = approxes > targets
    loss[is_bigger] = torch.abs(approxes[is_bigger] - targets[is_bigger])
    loss[~is_bigger] = torch.abs(approxes[~is_bigger] - targets[~is_bigger]) ** 4
    loss = loss.sum()
    return loss


X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

pytorch_mse_objective = CustomPytorchObjective(loss_function=MSELoss(reduction="sum"))
custom_objective = CustomPytorchObjective(loss_function=custom_loss)

for objective, objective_name in [
    ("RMSE", "Default RMSE"),
    (pytorch_mse_objective, "Pytorch MSE"),
    (custom_objective, "Pytorch Custom")]:
    model = CatBoostRegressor(
        loss_function=objective,
        n_estimators=200,
        eval_metric="RMSE",
        verbose=True)

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    plot_difference(pred_test, y_test, label=objective_name, is_show=False)

plt.legend()
plt.show()
