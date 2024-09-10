import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import json
import requests
import math
import io
import streamlit as st
from dashboard_fonctions import *

# importation du modèle
model_path = "./data/xgbclassifier.pkl"
model = load(model_path)

# importation des données
data = pd.read_csv('./data/df_test.csv')

# tri des IDs de clients
sorted_ids = sorted(data['SK_ID_CURR'].unique())

# importation des styles
with open("./styles/dashboard_style.css") as f:
    css = f.read()

css = f"{css}"
st.markdown(f"<style>{css}</style>", unsafe_allow_html = True)

st.markdown("""<div class="title">Simulation de Prêt Client</div>""", unsafe_allow_html=True)

# Utilisation de CSS pour appliquer un border-radius à l'image
st.markdown("""
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

st.markdown(f"""<div class="client-id">ID du client actuel : {int(current_client_id)}</div>""", unsafe_allow_html=True)

if st.button("Informations clients"):
    client_info_table = get_client_infos(current_client_id, data)
    
    if not client_info_table.empty:
        # Créez le code HTML pour la table
        html_table = """
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
    client_data = get_client_data(current_client_id, data)
    optimal_threshold = np.load('./data/optimal_threshold.npy').item()
    
    if client_data is not None:
        predictions = predict(client_data, model)
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
