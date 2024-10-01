import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

api_url = "https://pred-a-depenser-api.onrender.com/predict/"

# Importation des données
data = pd.read_csv('./data/df_test.csv')

# Tri des IDs de clients
sorted_ids = sorted(data['SK_ID_CURR'].unique())

# Importation des styles
with open("./styles/dashboard_style.css") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
st.markdown("<div class='title'>Simulation de Prêt Client</div>", unsafe_allow_html=True)

# Image
st.markdown("""
    <img class="styled-image" src="https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png" alt="Banner Image">
""", unsafe_allow_html=True)

# Initialisation de l'index de l'ID client
if 'current_id_index' not in st.session_state:
    st.session_state.current_id_index = 0

# Affichage de l'ID client actuel
current_client_id = sorted_ids[st.session_state.current_id_index]
selected_client_id = st.selectbox(
    'Sélectionnez l\'identifiant client :',
    options=sorted_ids,
    index=st.session_state.current_id_index,
    format_func=lambda x: f"Client {int(x)}"
)

if selected_client_id != current_client_id:
    st.session_state.current_id_index = sorted_ids.index(selected_client_id)
    current_client_id = selected_client_id


st.markdown(f"""<div class="client-id">ID du client actuel : {int(current_client_id)}</div>""", unsafe_allow_html=True)

# Lorsque l'utilisateur clique sur "Prédire"
if st.button("Simulation"):
    optimal_threshold = np.load('./data/optimal_threshold.npy').item()
    response = requests.get(f"{api_url}{int(selected_client_id)}")
    
    if response.status_code == 200:
        response_data = response.json()  # Charger le JSON de la réponse
        prediction = response_data.get('prediction')
        prediction_proba = response_data.get('proba')
        
        if prediction is not None and prediction_proba is not None:
            prediction_proba_percent = 100 * prediction_proba

            # Créer la jauge avec Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba_percent,
                number={'suffix': "%"},
                gauge={
                    'shape': "angular",
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black", 'showticklabels': True},
                    'bar': {'color': "green" if prediction_proba <= optimal_threshold else "red"},
                    'steps': [
                        {'range': [0, optimal_threshold * 100], 'color': "lightgreen"},
                        {'range': [optimal_threshold * 100, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': optimal_threshold * 100
                    }
                },
                domain={'x': [0, 1], 'y': [0, 1]},
            ))

            # Afficher la jauge dans Streamlit
            st.plotly_chart(fig)
                
            if prediction == 0:
                st.markdown("<h1 style='color:green; text-align:center;'>PRÊT ACCORDÉ</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='color:red; text-align:center;'>PRÊT REFUSÉ</h1>", unsafe_allow_html=True)
            
        else:
            st.error("Prédiction ou probabilité non trouvées dans la réponse.")
    else:
        st.error(f"Erreur API : {response.status_code} - {response.text}")
