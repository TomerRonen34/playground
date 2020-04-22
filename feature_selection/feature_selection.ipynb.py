#!/usr/bin/env python
# coding: utf-8

# # Exploration of differnt feature selection methods
# ### Based on "Census income classification with XGBoost" from the SHAP repository
# 
# Use xgboost version 0.9, otherwise you might get an error when using RFE feature selector:  
# pip install --upgrade xgboost==0.90

from sklearn.model_selection import train_test_split
import xgboost
import shap
import numpy as np
import matplotlib.pylab as pl
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from functools import partial

# print the JS visualization code to the notebook
shap.initjs()


# ## Load dataset

X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)


# # Sklearn feature statistics
# ### Use methods that estimate the statistical dependence between each feature (individually) and the target variable.
# 
# **chi2** - chi2 statistical test  
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
# 
# **mutual_info_classif** - estimation of the mutual information  
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html  
# 
# **f_classif** - ANOVA (analysis of variance)  
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html  
# Compares the differences between the mean value of the feature for each class, weighted by the variance of the feature inside each class.  
# I read a really good explanation that I can't find right now... This one is OK I guess:  
# https://towardsdatascience.com/anova-for-feature-selection-in-machine-learning-d9305e228476
# <img src="anova 75.png">

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


def calculate_multiple_feature_statistics(X, y, stat_functions):
    importance_df_list = []
    for stat_func in stat_functions:
        _importance_df = calculate_feature_statistics(stat_func, X, y)
        _importance_df = add_prefix_to_df_columns(_importance_df, stat_func.__name__)
        importance_df_list.append(_importance_df)

    importance_df = pd.concat(importance_df_list, axis="columns", sort=False)
    importance_df = order_df_columns_by_type(importance_df, columns_types=["rank", "stat", "p_values"])
    
    return importance_df


def calculate_feature_statistics(stat_func, X, y):
    feature_names = tuple(X.columns)
    
    stat_results = stat_func(X, y)

    p_values = None
    if isinstance(stat_results, tuple) and len(stat_results) == 2:
        importance_statistic, p_values = stat_results
    else:
        importance_statistic = stat_results

    importance_df = organize_feature_importances(importance_statistic, feature_names)

    if p_values is not None:
        importance_df = append_p_values(importance_df, p_values)
    
    return importance_df


def organize_feature_importances(statistic, feature_names):  
    stat_column = "stat"
    df = pd.DataFrame(zip(feature_names, statistic), columns=["feature_name", stat_column])
    df = df.sort_values(by=stat_column, ascending=False)
    df = df.set_index("feature_name")
    df["rank"] = range(1, len(df) + 1)
    return df


def append_p_values(importance_df, p_values):
    importance_df = importance_df.copy()
    importance_df["p_values"] = p_values
    return importance_df


def order_df_columns_by_type(df, columns_types):
    columns = pd.Series(df.columns)
    ordered_columns = []
    for columns_type in columns_types:
        curr_columns = columns[columns.str.contains(columns_type)].tolist()
        ordered_columns.extend(curr_columns)
    df = df[ordered_columns]
    return df


def add_prefix_to_df_columns(df, prefix):
    df = df.copy()
    prefixed_columns = [prefix + '_' + col for col in df.columns]
    df.columns = prefixed_columns
    return df
    

stat_functions = [f_classif, chi2, mutual_info_classif]
importance_df = calculate_multiple_feature_statistics(X, y, stat_functions)
importance_df


# # Using XGBoost to assign an importance value to each feature
# 
# **Using single-model feature importance**  
# Train a single XGBoost model using all features, calculate feature importances, and choose the most important features.  
# Supports both default xgboost feature importances and SHAP feature importances.
# 
# **Using univariate performance**  
# Split the train data into train and valiadtion, and train a different XGBoost model for each feature individually.  
# The estimated importance of the feature is the performance of its univariate model on the valiadtion set.

def xgboost_default_feature_importance(X, y):
    estimator = fit_classifier(X, y)
    feature_importances = estimator.feature_importances_
    return feature_importances


def xgboost_SHAP_importance(X, y):
    estimator = fit_classifier(X, y)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    feature_importances = np.mean(np.abs(shap_values), axis=0)
    return feature_importances


def xgboost_univariate_performance(X_trainval, y_trainval, metric="f1_macro"):
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=34)
    feature_names = X_train.columns
    performance = []
    for feature_name in feature_names:
        X_train_univariate = X_train[[feature_name]]
        X_val_univariate = X_val[[feature_name]]
        accuracy, f1 = fit_and_eval_classifier(X_train_univariate, y_train, X_val_univariate, y_val, n_estimators=50)
        if metric == "f1_macro":
            _performance = f1
        else:
            _performance = accuracy
        performance.append(_performance)
    return performance


def fit_and_eval_classifier(X_train, y_train, X_test, y_test, n_estimators=200):
    estimator = fit_classifier(X_train, y_train, n_estimators)
    accuracy, f1 = eval_classifier(estimator, X_test, y_test)
    return accuracy, f1


def fit_classifier(X, y, n_estimators=200):
    fit_params = {
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y),
        "eval_metric": "logloss",
        "n_estimators": n_estimators,
        "seed": 34,
        "random_state": 34
    }
    estimator = XGBClassifier(**fit_params)
    estimator.fit(X, y)
    return estimator


def eval_classifier(estimator, X, y):
    pred_labels = estimator.predict(X)
    accuracy = accuracy_score(pred_labels, y)
    f1 = f1_score(pred_labels, y, average="macro")
    return accuracy, f1


stat_functions = [xgboost_default_feature_importance, xgboost_SHAP_importance, xgboost_univariate_performance]
importance_df = calculate_multiple_feature_statistics(X, y, stat_functions)
importance_df


# # RFE - Recursive method based on feature importances
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# 
# A method similar to Sequential Feature Selection (but not quite the same).  
# Starting with all the features, this method trains a model and calculate feature importances. Then, the least important feature is eliminated, and the method continues to train models on smaller and smaller subsets of features, until it reaches the desired amount.
# 
# Since sklearn's RFE implementation uses the model's "feature_importances_" property, a variant of XGBClassifier is implemented to provide in-class support of SHAP importances.
# 
# A version of this method that uses cross-validation also exists (RFECV) but not used in this notebook.

import numpy as np
import shap
from xgboost import XGBClassifier


class ShapXGBClassifier(XGBClassifier):
    """
    A version of XGBClassifier that uses SHAP feature importances
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vanilla_model = XGBClassifier(**kwargs)
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


from sklearn.feature_selection import RFE

def select_features_with_RFE(estimator, X, y, top_k):
    feature_names = X.columns.values
    selector = RFE(estimator, n_features_to_select=top_k, step=1)
    selector = selector.fit(X, y)
    selected_features_inds = np.nonzero(selector.support_)[0]
    selected_features = feature_names[selected_features_inds]
    return selected_features


RFE_fit_params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y),
    "eval_metric": "logloss",
    "n_estimators": 50,
    "seed": 34,
    "random_state": 34
}

selected_features = select_features_with_RFE(estimator=XGBClassifier(**RFE_fit_params), X=X_train, y=y_train, top_k=2)
print("RFE with default xgboost importances:", selected_features)

selected_features = select_features_with_RFE(estimator=ShapXGBClassifier(**RFE_fit_params), X=X_train, y=y_train, top_k=2)
print("RFE with SHAP importances:", selected_features)


# # Compare methods
# ### # TODO: Add sequential feature selection from mlxtend (forward and backward)

def bulk_choose_features(X, y, stat_func, top_k):
    importance_df = calculate_feature_statistics(stat_func, X_train, y_train)
    feature_names_by_importance = importance_df["rank"].sort_values().index.tolist()
    selected_features = feature_names_by_importance[:top_k]
    return selected_features


def compare_methods(X_train, y_train, X_test, y_test):
    print("\n\n")
    print("compare_methods")
    print("===============")
    print("full model")
    accuracy_full, f1_full = fit_and_eval_classifier(X_train, y_train, X_test, y_test)
    results_columns = ["method", "top_k_features", "f1_macro", "accuracy", "selected_features"]
    results_rows = []
    results_rows.append(("no selection", X.shape[1], f1_full, accuracy_full, ["all features"]))
    
    bulk_stat_functions = [xgboost_SHAP_importance, xgboost_default_feature_importance,
                      xgboost_univariate_performance, f_classif, chi2, mutual_info_classif]
    bulk_names = [stat_func.__name__ for stat_func in bulk_stat_functions]
    bulk_choice_functions = [partial(bulk_choose_features, stat_func=stat_func) for stat_func in bulk_stat_functions]
    
    RFE_names = ["RFE with default xgboost importances", "RFE with SHAP importances"]
    RFE_estimators = [XGBClassifier(**RFE_fit_params), ShapXGBClassifier(**RFE_fit_params)]
    RFE_choice_functions = [partial(select_features_with_RFE, estimator=estimator) for estimator in RFE_estimators]
    
    method_names =  RFE_names + bulk_names
    choice_functions = RFE_choice_functions + bulk_choice_functions
    
    for method_name, choice_function in zip(method_names, choice_functions):
        for top_k in [2, 5]:
            print(method_name, top_k)
            selected_features = choice_function(X=X_train, y=y_train, top_k=top_k)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            accuracy, f1 = fit_and_eval_classifier(X_train_selected, y_train, X_test_selected, y_test)
            results_rows.append((method_name, top_k, f1, accuracy, selected_features))
    results = pd.DataFrame(results_rows, columns=results_columns)
    results = results.sort_values(by="f1_macro", ascending=False)
    results.index = range(1, len(results) + 1)
    return results


results = compare_methods(X_train, y_train, X_test, y_test)
results

