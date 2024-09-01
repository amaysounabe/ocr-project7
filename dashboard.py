import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import json
import requests
import math
import io

# URL de l'endpoint d'invocation du modèle MLflow
#MLFLOW_MODEL_URI = "http://127.0.0.1:6000/invocations"

model_path = "best_model/xgbclassifier.pkl"
model = load(model_path)

#importation des données
data = pd.read_csv('df_test.csv')

# Tri des IDs de clients
sorted_ids = sorted(data['SK_ID_CURR'].unique())

# Fonction pour obtenir les données du client en fonction de l'ID
def get_client_data(current_client_id):
    client_data = data[data['SK_ID_CURR'] == current_client_id]
    if client_data.empty:
        return None
    # Retourne les features du client (sans l'ID)
    return client_data.drop(columns=['SK_ID_CURR']).values.tolist()


#fonction pour récupérer les infos du client
def get_client_infos(current_client_id):
    client_data = data[data['SK_ID_CURR'] == current_client_id]
    if not client_data.empty:
        age_client = math.floor(client_data['DAYS_BIRTH'].values[0] / 365)
        nb_children_client = round(client_data['CNT_CHILDREN'].values[0])
        income_client = round(client_data['AMT_INCOME_TOTAL'].values[0])
        client_info_table = pd.DataFrame({
            'Caractéristiques' : ['Âge', 'Nombre d\'enfants','Revenus totaux'],
            'Données' : [age_client, nb_children_client, income_client]
        })
        return client_info_table
    
# Fonction pour envoyer la requête au modèle MLflow
def predict(input_data):
    
    try:    
        predictions = model.predict_proba(input_data)
        return predictions
    except Exception as e:
        st.error(f"Erreur rencontrée lors de la prédiction : {e}")
        return None

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bree+Serif&display=swap');
        .title {
            font-family: 'Bree Serif', serif;                /* Changer la police */
            font-weight: bold;
            font-size: 48px;                   /* Changer la taille de la police */
            color: #f0f0f0;                    /* Changer la couleur du texte (vert) */
            text-align: center;                /* Centrer le texte */
            border: 2px solid #f0f0f0;              /* Bordure autour du texte */
            padding: 15px;                         /* Espacement autour du texte */
            border-radius: 10px;                   /* Coins arrondis pour la bordure */
            background-color: #1e1e1e;             /* Couleur de fond du texte */
            margin-bottom: 50px;               /* Ajouter un espace en bas */
        }
        .client-id {
            width: 100%;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            padding: 10px;
            border: 2px solid #ffffff;
            border-radius: 10px;
            background-color: #120D16;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px; /* Espacement entre les boutons */
            margin-top: 20px;
        }
    </style>
    <div class="title">Simulation de Prêt Client</div>
""", unsafe_allow_html=True)

# Utilisation de CSS pour appliquer un border-radius à l'image
st.markdown(f"""
    <style>
        .styled-image {{
            border-radius: 15px;  /* Arrondir les coins de l'image */       
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 30px;
        }}
    </style>
    <img class="styled-image" src="https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png" alt="Banner Image">
""", unsafe_allow_html=True)


# Initialisation de l'index de l'ID client dans le session_state
if 'current_id_index' not in st.session_state:
    st.session_state.current_id_index = 0

# Affichage de l'ID client actuel
current_client_id = sorted_ids[st.session_state.current_id_index]

selected_client_id = st.selectbox(
    'Sélectionnez l\'identifiant client :',
    options = sorted_ids,
    index = st.session_state.current_id_index,
    format_func = lambda x:f"Client {int(x)}"
)

# Mise à jour de l'index actuel si l'utilisateur sélectionne un ID dans le selectbox
if selected_client_id != current_client_id:
    st.session_state.current_id_index = sorted_ids.index(selected_client_id)
    current_client_id = selected_client_id

st.markdown(f"""
<div class='client-id'>
    <h2>ID du client actuel : <span style='color:yellow;'>{int(current_client_id)}</span></h2>
</div>""", unsafe_allow_html=True)

if st.button("Informations clients"):
    client_info_table = get_client_infos(current_client_id)
    
    if not client_info_table.empty:
        # Créez le code HTML pour la table
        html_table = """
        <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
                overflow: hidden;
                margin-bottom: 50px;
                border: none;
                border-color: #ddd; /* Couleur de la bordure extérieure */
            }
            .custom-table th, .custom-table td {
                border: 1px solid #ffffff;
                padding: 8px;
                text-align: left;
                color : #ffffff;
            }
            .custom-table th {
                background-color: #111111;
                font-weight: bold;
                color: #ffffff;
            }
            .custom-table th:first-child {
                text-align: left;  /* Aligner le titre de la première colonne à gauche */
            }
            .custom-table th:last-child {
                text-align: right;  /* Aligner le titre de la dernière colonne à droite */
            }
            .custom-table td {
                text-align: right; /* Alignement du texte à droite pour les valeurs */
            }
            .custom-table td:first-child {
                text-align: left; /* Alignement à gauche pour la première colonne */
            }
            .custom-table td:last-child {
                text-align: right; /* Alignement à droite pour la dernière colonne */
            }
            .custom-table tr:nth-child(odd) {
                background-color: #090829; /* Couleur de fond des lignes impaires en noir */
            }
            .custom-table tr:nth-child(even) {
                background-color: #120D16; /* Couleur de fond des lignes paires en gris foncé */
            }
            .custom-table tr:hover {
                background-color: #aaaaaa;
            }
            .table-container {
                border: none; /* Supprimer la bordure extérieure du conteneur */
                padding: 0; /* Supprimer le padding pour éviter les bordures supplémentaires */
                margin-bottom: 50px;
                margin-top: 20px;
        </style>
        <div class = "table-container">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Caractéristiques</th>
                        <th>Valeurs</th>
                    </tr>
                </thead>   
                <tbody>"""
        for i in client_info_table.index:
            line = f"""
            <tr>
                <td>{client_info_table.iloc[i, 0]}</td>
                <td>{client_info_table.iloc[i, 1]}</td>
            </tr>"""
            html_table = html_table + line
        html_table = html_table + """
                </tbody>
            </table>
        </div>
        """
        
        # Afficher la table HTML
        st.markdown(html_table, unsafe_allow_html=True)

    else:
        st.write(f"Données non renseignées.")
    

# Lorsque l'utilisateur clique sur "Prédire"
if st.button("Simulation"):
    client_data = get_client_data(current_client_id)
    optimal_threshold = np.load('./best_model/optimal_threshold.npy').item()
    
    if client_data is not None:
        predictions = predict(client_data)
        if predictions is not None:
            prediction_value = predictions[0, 1]
            prediction_value_percent = 100 * prediction_value

            # Créer la jauge avec Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_value_percent,
                number = {'suffix': "%"},  # Ajoute le symbole % à la valeur
                gauge = {
                    'shape': "angular",  # Forme angulaire pour une jauge semi-circulaire
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black", 'showticklabels': True},
                    'bar': {'color': "green" if prediction_value <= optimal_threshold else "red"},  # Couleur conditionnelle
                    'steps': [
                        {'range': [0, optimal_threshold * 100], 'color': "lightgreen"},
                        {'range': [optimal_threshold * 100, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': optimal_threshold * 100 # Afficher le seuil optimal
                    }
                },
                domain = {'x': [0, 1], 'y': [0, 1]},
                #title = {'text': "Probabilité de Prêt Accordé"}
            ))

            # Afficher la jauge dans Streamlit
            st.plotly_chart(fig)
            
            if prediction_value <= optimal_threshold:
                st.markdown("<h1 style='color:green; text-align:center;'>PRÊT ACCORDÉ</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='color:red; text-align:center;'>PRÊT REFUSÉ</h1>", unsafe_allow_html=True)
        
    else:
        st.error("ID client non trouvé dans les données.")
