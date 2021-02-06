import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from catboost import CatBoostRegressor
from catboost import Pool
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import MSELoss
from pathlib import Path

from pytorch_catboost.custom_pytorch_objective import CustomPytorchObjective


def plot_relative_difference(preds, y, label="", bins=15, title=None, is_show=True):
    rel_diff = _relative_difference(preds, y)
    _, ret_bins, _ = plt.hist(rel_diff, bins=bins, alpha=0.4, label=label)

    if title is not None:
        plt.title(title)
    if is_show:
        plt.legend()
        plt.show()

    return ret_bins


def _relative_difference(preds, y):
    rel_diff = (preds - y) / y
    return rel_diff


def custom_loss(preds, targets):
    raw_diff = preds - targets
    loss = _custom_loss(raw_diff)
    loss = loss.sum()
    return loss


def _custom_loss(raw_diff):
    """  raw_diff = preds - targets  """
    is_undershoot = raw_diff <= 0
    loss = torch.zeros_like(raw_diff)
    loss[is_undershoot] = torch.abs(raw_diff[is_undershoot] - 1) ** 3 - 1
    loss[~is_undershoot] = torch.abs(raw_diff[~is_undershoot])
    return loss


def custom_regression_loss(preds: Tensor, targets: Tensor) -> Tensor:
    raw_diff = preds - targets
    is_undershoot = raw_diff < 0
    loss = torch.zeros_like(raw_diff)
    loss[is_undershoot] = torch.abs(raw_diff[is_undershoot] - 1) ** 3 - 1
    loss[~is_undershoot] = torch.abs(raw_diff[~is_undershoot])
    loss = loss.sum()
    return loss


def main():
    exp_dir = Path("experiments/regression")
    exp_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_pool = Pool(X_train, y_train, weight=np.random.rand(len(X_train)))
    # train_pool = Pool(X_train, y_train)

    pytorch_mse_objective = CustomPytorchObjective(loss_function=MSELoss(reduction="sum"))
    custom_objective = CustomPytorchObjective(loss_function=custom_loss)

    n_estimators = 300
    results = {"y_test": y_test,
               "n_estimators": n_estimators,
               "pred_test": {}}

    bins = None
    for objective, objective_name in [
        (custom_objective, "Custom Pytorch Objective"),
        ("RMSE", "Default RMSE"),
        # (pytorch_mse_objective, "Pytorch MSE")
    ]:
        model = CatBoostRegressor(
            loss_function=objective,
            n_estimators=n_estimators,
            eval_metric="RMSE",
            verbose=True)

        # model.fit(train_pool)
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)

        results["pred_test"][objective_name] = pred_test

    results_path = exp_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    plot_results(results_path)


def plot_results(results_path: Path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    y_test = results["y_test"]

    all_rel_diff = [
        _relative_difference(preds, y_test)
        for preds in results["pred_test"].values()
    ]
    all_rel_diff = np.hstack(all_rel_diff)

    min_value = all_rel_diff.min()
    max_value = all_rel_diff.max()
    bin_size = np.abs(min_value) / 3
    # bins = np.arange(min_value-1, max_value + 1+1e-4, bin_size)
    bins = np.arange(min_value, max_value + bin_size, bin_size)

    for objective_name in reversed(list(results["pred_test"].keys())):
        pred_test = results["pred_test"][objective_name]
        plot_relative_difference(pred_test, y_test, label=objective_name, bins=bins, is_show=False)

    plt.axvline(0, linestyle='--', color='k', alpha=0.5)
    plt.legend()
    plt.xlabel("relative difference = (preds - targets) / targets")
    plt.show()


if __name__ == '__main__':
    # main()
    # plot_results("experiments/regression/results_200.pkl")
    plot_results("experiments/regression/results_300.pkl")
