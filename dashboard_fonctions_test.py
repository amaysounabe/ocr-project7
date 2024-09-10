import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dashboard_fonctions import *

# création de données factices
@pytest.fixture
def mock_data():
    data = pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3],
        'DAYS_BIRTH': [10000, 20000, 30000],
        'CNT_CHILDREN': [2, 1, 0],
        'AMT_INCOME_TOTAL': [50000, 60000, 70000]
    })
    return data


# test pour la fonction get_client_data
def test_get_client_data(mock_data):
    client_id = 1
    expected_data = [[10000, 2, 50000]]
    result = get_client_data(client_id, mock_data)
    assert result == expected_data


# test pour la fonction get_client_infos
def test_get_client_infos(mock_data):
    client_id = 1
    expected_table = pd.DataFrame({
        'Caractéristiques': ['Âge', 'Nombre d\'enfants', 'Revenus totaux'],
        'Données': [27, 2, 50000]
    })
    result = get_client_infos(client_id, mock_data)
    pd.testing.assert_frame_equal(result, expected_table)


# on entraine un modele simple
@pytest.fixture
def mock_model():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


# on teste la fonction predict
def test_predict(mock_model):
    input_data = np.array([[1, 2, 3]])
    predictions = predict(input_data, mock_model)
    assert predictions is not None
    assert predictions.shape[1] == 2