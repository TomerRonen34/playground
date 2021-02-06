from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn import BCEWithLogitsLoss, SoftMarginLoss
from pytorch_catboost.custom_pytorch_objective import CustomPytorchObjective

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)


def soft_margin_loss(approxes, targets):
    targets = 2 * targets - 1
    return SoftMarginLoss()(approxes, targets)


model = CatBoostClassifier(
    loss_function=CustomPytorchObjective(loss_function=soft_margin_loss),
    n_estimators=50,
    eval_metric="Accuracy",
    verbose=True)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))

a = 0
