from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn import BCEWithLogitsLoss
import matplotlib.pyplot as plt
import numpy as np

from pytorch_catboost.pytorch_second_derivative import minus_bce_loss, \
    basic_dml_loss, margin_dml_loss
from pytorch_catboost.patbas_objectives import PatbasLoglossObjective
from pytorch_catboost.custom_pytorch_objective import CustomPytorchObjective

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# model = CatBoostClassifier(loss_function=LoglossObjective(), eval_metric="Accuracy", verbose=True)
# model = CatBoostClassifier(loss_function=CustomPytorchObjective(loss_function=minus_bce_loss), eval_metric="Accuracy", verbose=True)
# model = CatBoostClassifier(loss_function=CustomPytorchObjective(loss_function=BCEWithLogitsLoss(reduction="sum")), eval_metric="Accuracy", verbose=True)

# model = CatBoostRegressor(loss_function=CustomPytorchObjective(loss_function=basic_dml_loss),
#                           n_estimators=1000,
#                           eval_metric="Accuracy",
#                           verbose=True)

model = CatBoostRegressor(loss_function=CustomPytorchObjective(loss_function=margin_dml_loss),
                          n_estimators=200,
                          eval_metric="Accuracy",
                          verbose=True)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)

pred0 = pred_train[y_train == 0] ** 2
pred0 = pred0[(pred0 > np.quantile(pred0, 0.1)) & (pred0 < np.quantile(pred0, 0.9))]
plt.hist(pred0, label="0", bins=10, alpha=0.5)
plt.title("train 0")
plt.show()

pred1 = pred_train[y_train == 1] ** 2
pred1 = pred1[(pred1 > np.quantile(pred1, 0.1)) & (pred1 < np.quantile(pred1, 0.9))]
plt.hist(pred1, label="1", bins=10, alpha=0.5)
plt.title("train 1")
plt.show()

pred_test = model.predict(X_test)

pred0 = pred_test[y_test == 0] ** 2
pred0 = pred0[(pred0 > np.quantile(pred0, 0.1)) & (pred0 < np.quantile(pred0, 0.9))]
plt.hist(pred0, label="0", bins=10, alpha=0.5)
plt.title("test 0")
plt.show()

pred1 = pred_test[y_test == 1] ** 2
pred1 = pred1[(pred1 > np.quantile(pred1, 0.1)) & (pred1 < np.quantile(pred1, 0.9))]
plt.hist(pred1, label="1", bins=10, alpha=0.5)
plt.title("test 1")
plt.legend()
plt.show()

print(classification_report(y_train, pred_train ** 2 < 50))
print(classification_report(y_test, pred_test ** 2 < 50))

a = 0
