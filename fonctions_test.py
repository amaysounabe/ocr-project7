import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from fonctions import score_metier, mlflow_run_model, feature_importance, threshold_optimization


# Data pour les tests
np.random.seed(0)
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Fonction pour tester score_metier
def test_score_metier():
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    fp_coeff = 1
    fn_coeff = 10
    score = score_metier(y_test, y_pred, fp_coeff, fn_coeff)
    assert isinstance(score, float)
    assert 0 <= score

# Fonction pour tester mlflow_run_model
def test_mlflow_run_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    params = {'C': [0.1, 1, 10]}
    metric = 'accuracy'
    n_folds = 2
    artifact_path = 'logistic_regression'
    registered_model_name = 'LogisticRegressionModel'

    # Test mlflow_run_model
    df_params, df_scores = mlflow_run_model(
        experiment_name="Test Experiment",
        model=model,
        params=params,
        metric=metric,
        n_folds=n_folds,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )

    assert isinstance(df_params, pd.DataFrame)
    assert isinstance(df_scores, pd.DataFrame)
    assert not df_params.empty
    assert not df_scores.empty

#gestion des inputs
@patch('builtins.input', return_value = 'n')

# Fonction pour tester feature_importance
def test_feature_importance(mock_input):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    columns_list = [f'feature_{i}' for i in range(X.shape[1])]
    data_features_importance, data_shap = feature_importance(model, X_train, X_test, columns_list, is_linear=True)

    assert isinstance(data_features_importance, pd.DataFrame)
    assert isinstance(data_shap, pd.DataFrame)
    assert not data_features_importance.empty
    assert not data_shap.empty


#gestion des inputs
@patch('builtins.input', return_value = 'n')

# Fonction pour tester threshold_optimization
def test_threshold_optimization(mock_input):
    y_proba = np.random.rand(100)
    thresholds_range = np.linspace(0, 1, 20)

    # On utilise des valeurs arbitraires pour les coûts ici
    best_threshold = threshold_optimization(y, y_proba, thresholds_range)
    
    assert isinstance(best_threshold, float)
    assert 0 <= best_threshold <= 1