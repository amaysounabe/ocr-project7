<h1 style="text-align: center; font-size: 35px;">OPENCLASSROOMS - PROJET 7 <br> 
    Implémentation d'un modèle de scoring</h1>



<div style = "text-align: center;">
    <img src = "https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png", alt="Bannière" style="width: 70%">
</div>

## Description
La société **Prêt à dépenser** propose des crédits à la consommation pour des 
personnes ayant peu ou pas d'historiques de prêt. <br>
- La page OpenClassRooms du projet est disponible [**ici**](https://openclassrooms.com/fr/paths/793/projects/1504)
- Les données utilisées au cours de ce projet sont disponibles [**ici**](https://www.kaggle.com/c/home-credit-default-risk/data) ou peuvent être téléchargées via la commande suivante
    ```bash
    pip install kaggle
    kaggle competitions download -c home-credit-default-risk
    ```

### Objectifs
- Mise en oeuvre d'un outil de **scoring crédit** qui calcule la probabilité qu'un client rembourse son crédit
- Mise en production du modèle de classification à l'aide d'une **API**
- Détection de dérives de données

### Fichiers
- **notebook_modelisation.ipynb** &rarr; Nootebook comportant : EDA, feature engineering, modélisation & optimisation du modèle final
- **fonctions.py** &rarr; Fichier python comportant les fonctions utilisées dans le notebook
- **dashboard_interface.py** &rarr; Fichier python comportant la configuration du dashboard **Streamlit**
- **dashboard_fonctions.py** &rarr; Fichier python comportant les fonctions utilisées dans le dashboard
- **fonctions_test.py** &rarr; Fichier python comportant les tests unitaires des fonctions contenues dans **fonctions.py**
- **dashboard_fonctions_test.py** &rarr; Fichier comportant les tests unitaires des fonctions contenues dans **dashboard_fonctions.py**
- **requirements.txt** &rarr; Fichier texte contenant les librairies python nécessaires

### Détails
- Version de Python utilisée &rarr; **Python 3.12**
- Liste des librairies utilisées
    - **numpy**
    - **pandas**
    - **matplotlib**
    - **seaborn**
    - **scikit-learn**
    - **xgboost**
    - **mlflow**
    - **shap**
    - **joblib**
    - **evidently**
    - **pytest**
    - **streamlit**

## Procédure
- Analyse exploratoire des données (à l'aide d'un  moteur [**KAGGLE**](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook)
- Feature engineering
- Entraînement des modèles de classification et tracking via **Mlflow**
- Optimisation du meilleur modèle via un **score** adapté au contexte métier
- &Eacute;tude de l'importance des features via **shap**
- Réalisation du dashboard **Streamlit**
- Déploiement du dashboard sur le cloud
- &Eacute;tude de la dérive de données

## Dashboard interactif
Le dashboard **Streamlit** en production est disponible [**ici**](https://mayad-ocr-pret-a-depenser.streamlit.app/)

## Installation
Pour une utilisation locale du dashboard interactif, tapez les commandes suivantes dans votre terminal

```bash
git clone https://github.com/amaysounabe/ocr-project7.git
cd ocr-project7
pip install -r requirements.txt
streamlit run dashboard_interface.py
