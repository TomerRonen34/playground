#!/usr/bin/env python
# coding: utf-8

# # NHANES I Survival Model
# 
# This is a cox proportional hazards model on data from <a href="https://wwwn.cdc.gov/nchs/nhanes/nhanes1">NHANES I</a> with followup mortality data from the <a href="https://wwwn.cdc.gov/nchs/nhanes/nhefs">NHANES I Epidemiologic Followup Study</a>. It is designed to illustrate how SHAP values enable the interpretion of XGBoost models with a clarity traditionally only provided by linear models. We see interesting and non-linear patterns in the data, which suggest the potential of this approach. Keep in mind the data has not yet been checked by us for calibrations to current lab tests and so you should not consider the results as actionable medical insights, but rather a proof of concept. 
# 
# Note that support for Cox loss and SHAP interaction effects were only recently merged, so you will need the latest master version of XGBoost to run this notebook.

import shap
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pylab as pl


# ## Prepare Data
# 
# This uses a pre-processed subset of NHANES I data available in the SHAP datasets module.

X,y = shap.datasets.nhanesi()
X_display,y_display = shap.datasets.nhanesi(display=True) # human readable feature values


# ## Vanilla XGBoost model

fit_params = {
    "eta": 0.002,
    "max_depth": 3, 
    "objective": "survival:cox",
    "subsample": 0.5,
    "n_estimators": 100,
    "random_state": 34,
    "seed": 34
}
orig_model = XGBRegressor(**fit_params)
orig_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=int(fit_params["n_estimators"]/5))


# ## XGBoost model with SHAP feature importances

import numpy as np
import shap
from xgboost import XGBRegressor


class ShapXGBRegressor(XGBRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vanilla_model = XGBRegressor(**kwargs)
        self.feature_importances = None

    @property
    def feature_importances_(self):
        return self.feature_importances
    
    @property
    def vanilla_feature_importances(self):
        return super().feature_importances_

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.vanilla_model.__dict__ = self.__dict__
        self._calculate_feature_importances(X)
        return self

    def _calculate_feature_importances(self, X):
        explainer = shap.TreeExplainer(self.vanilla_model)
        shap_values = explainer.shap_values(X)
        feature_importances = np.mean(np.abs(shap_values), axis=0)
        self.feature_importances = feature_importances

    def get_params(self, deep=False):
        return self.vanilla_model.get_params(deep)


my_model = ShapXGBRegressor(**fit_params)
my_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=int(fit_params["n_estimators"]/5))


# ## Compare feature importances

(my_model.vanilla_feature_importances == orig_model.feature_importances_).all()


orig_model.feature_importances_


my_model.feature_importances_


# ## Use models for feature selection

from sklearn.impute import SimpleImputer

def impute(df):
    imputer = SimpleImputer(strategy='median')
    df_imputed = df.copy()
    df_imputed[:] = imputer.fit_transform(df)
    return df_imputed    

def impute_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    
    X_train_imputed = X_train.copy()
    X_train_imputed[:] = imputer.transform(X_train)
    
    X_test_imputed = X_test.copy()
    X_test_imputed[:] = imputer.transform(X_test)
    
    return X_train_imputed, X_test_imputed

X_train_imputed, X_test_imputed = impute_train_test(X_train, X_test)
X_imputed = impute(X)


from sklearn.feature_selection import RFE

estimator = XGBRegressor(**fit_params)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X_imputed, y)
print(np.nonzero(selector.support_)[0])
print(selector.support_)
print(selector.ranking_)

estimator = ShapXGBRegressor(**fit_params)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X_imputed, y)
print(np.nonzero(selector.support_)[0])
print(selector.support_)
print(selector.ranking_)


np.argsort(orig_model.feature_importances_)[::-1]


np.argsort(my_model.feature_importances_)[::-1]


X.columns[np.nonzero(selector.support_)[0]]


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = ShapXGBRegressor(**params, random_state=34, seed=34)
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)

X = X[:,::-1]
estimator = ShapXGBRegressor(**params, random_state=34, seed=34)
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)


# ## Explain the model's predictions on the entire dataset

shap_values = shap.TreeExplainer(model).shap_values(X)


# ### SHAP Summary Plot
# 
# The SHAP values for XGBoost explain the margin output of the model, which is the change in log odds of dying for a Cox proportional hazards model. We can see below that the primary risk factor for death according to the model is being old. The next most powerful indicator of death risk is being a man.
# 
# This summary plot replaces the typical bar chart of feature importance. It tells which features are most important, and also their range of effects over the dataset. The color allows us match how changes in the value of a feature effect the change in risk (such that a high white blood cell count leads to a high risk of death).

shap.summary_plot(shap_values, X)


# ### SHAP Dependence Plots
# 
# While a SHAP summary plot gives a general overview of each feature a SHAP dependence plot show how the model output varies by feauture value. Note that every dot is a person, and the vertical dispersion at a single feature value results from interaction effects in the model. The feature used for coloring is automatically chosen to highlight what might be driving these interactions. Later we will see how to check that the interaction is really in the model with SHAP interaction values. Note that the row of a SHAP summary plot results from projecting the points of a SHAP dependence plot onto the y-axis, then recoloring by the feature itself.
# 
# Below we give the SHAP dependence plot for each of the NHANES I features, revealing interesting but expected trends. Keep in mind the calibration of some of these values can be different than a modern lab test so be careful drawing conclusions.

# we pass "Age" instead of an index because dependence_plot() will find it in X's column names for us
# Systolic BP was automatically chosen for coloring based on a potential interaction to check that 
# the interaction is really in the model see SHAP interaction values below
shap.dependence_plot("Age", shap_values, X)


# we pass display_features so we get text display values for sex
shap.dependence_plot("Sex", shap_values, X)


# setting show=False allows us to continue customizing the matplotlib plot before displaying it
shap.dependence_plot("Systolic BP", shap_values, X, show=False)
pl.xlim(80,225)
pl.show()


shap.dependence_plot("Poverty index", shap_values, X)


shap.dependence_plot("White blood cells", shap_values, X, display_features=X_display, show=False)
pl.xlim(2,15)
pl.show()


shap.dependence_plot("BMI", shap_values, X, display_features=X_display, show=False)
pl.xlim(15,50)
pl.show()


shap.dependence_plot("Serum magnesium", shap_values, X, show=False)
pl.xlim(1.2,2.2)
pl.show()


shap.dependence_plot("Sedimentation rate", shap_values, X)


shap.dependence_plot("Serum protein", shap_values, X)


shap.dependence_plot("Serum cholesterol", shap_values, X, show=False)
pl.xlim(100,400)
pl.show()


shap.dependence_plot("Pulse pressure", shap_values, X)


shap.dependence_plot("Serum iron", shap_values, X, display_features=X_display)


shap.dependence_plot("TS", shap_values, X)


shap.dependence_plot("Red blood cells", shap_values, X)


# ## Compute SHAP Interaction Values
# 
# See the Tree SHAP paper for more details, but briefly, SHAP interaction values are a generalization of SHAP values to higher order interactions. Fast exact computation of pairwise interactions are implemented in the latest version of XGBoost with the pred_interactions flag. With this flag XGBoost returns a matrix for every prediction, where the main effects are on the diagonal and the interaction effects are off-diagonal. The main effects are similar to the SHAP values you would get for a linear model, and the interaction effects captures all the higher-order interactions are divide them up among the pairwise interaction terms. Note that the sum of the entire interaction matrix is the difference between the model's current output and expected output, and so the interaction effects on the off-diagonal are split in half (since there are two of each). When plotting interaction effects the SHAP package automatically multiplies the off-diagonal values by two to get the full interaction effect.

# takes a couple minutes since SHAP interaction values take a factor of 2 * # features
# more time than SHAP values to compute, since this is just an example we only explain
# the first 2,000 people in order to run quicker
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X.iloc[:2000,:])


# ### SHAP Interaction Value Summary Plot
# 
# A summary plot of a SHAP interaction value matrix plots a matrix of summary plots with the main effects on the diagonal and the interaction effects off the diagonal.

shap.summary_plot(shap_interaction_values, X.iloc[:2000,:])


# ### SHAP Interaction Value Dependence Plots
# 
# Running a dependence plot on the SHAP interaction values a allows us to separately observe the main effects and the interaction effects.
# 
# Below we plot the main effects for age and some of the interaction effects for age. It is informative to compare the main effects plot of age with the earlier SHAP value plot for age. The main effects plot has no vertical dispersion because the interaction effects are all captured in the off-diagonal terms.

shap.dependence_plot(
    ("Age", "Age"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


# Now we plot the interaction effects involving age. These effects capture all of the vertical dispersion that was present in the original SHAP plot but is missing from the main effects plot above. The plot below involving age and sex shows that the sex-based death risk gap varies by age and peaks at age 60.

shap.dependence_plot(
    ("Age", "Sex"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Age", "Systolic BP"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Age", "White blood cells"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Age", "Poverty index"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Age", "BMI"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Age", "Serum magnesium"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


# Now we show a couple examples with systolic blood pressure.

shap.dependence_plot(
    ("Systolic BP", "Systolic BP"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Systolic BP", "Age"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


shap.dependence_plot(
    ("Systolic BP", "Age"),
    shap_interaction_values, X.iloc[:2000,:],
    display_features=X_display.iloc[:2000,:]
)


import matplotlib.pylab as pl
import numpy as np


tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
pl.figure(figsize=(12,12))
pl.imshow(tmp2)
pl.yticks(range(tmp2.shape[0]), X.columns[inds], rotation=50.4, horizontalalignment="right")
pl.xticks(range(tmp2.shape[0]), X.columns[inds], rotation=50.4, horizontalalignment="left")
pl.gca().xaxis.tick_top()
pl.show()

