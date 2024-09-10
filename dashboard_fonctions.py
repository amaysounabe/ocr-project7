# Fonctions utilisées pour le dashboard Streamlit

# fonction pour obtenir les données du client en fonction de l'ID
def get_client_data(current_client_id, data):
    client_data = data[data['SK_ID_CURR'] == current_client_id]
    if client_data.empty:
        return None
    # Retourne les features du client (sans l'ID)
    return client_data.drop(columns=['SK_ID_CURR']).values.tolist()


# fonction pour récupérer les infos du client
def get_client_infos(current_client_id, data):
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


# fonction pour envoyer la requête au modèle MLflow
def predict(input_data, model):   
    try:    
        predictions = model.predict_proba(input_data)
        return predictions
    except Exception as e:
        st.error(f"Erreur rencontrée lors de la prédiction : {e}")
        return None