import os
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.models.evaluation import evaluate
import shap
import gc

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier





def score_metier(y_test, y_pred, fp_coeff, fn_coeff):

    """
    Calcule un score métier pour évaluer la performance d'un modèle de classification binaire en fonction des coûts associés aux faux positifs et aux faux négatifs.

    Le score est calculé en tenant compte des coûts spécifiques pour les faux positifs (FP) et les faux négatifs (FN). Le coût réel est comparé au coût maximal possible, puis le score est inversé pour que les meilleurs modèles aient des scores plus proches de 1.

    ** Paramètres : 
    - `y_test` (array-like): Les vraies étiquettes des données de test. Ces valeurs représentent les classes réelles des instances dans l'ensemble de test.
    - `y_pred` (array-like): Les étiquettes prédites par le modèle pour les données de test. Ces valeurs représentent les classes prédites par le modèle pour chaque instance.
    - `fp_coeff` (float): Le coefficient de coût associé aux faux positifs (FP). Cela représente le coût unitaire d'un faux positif.
    - `fn_coeff` (float): Le coefficient de coût associé aux faux négatifs (FN). Cela représente le coût unitaire d'un faux négatif.

    ** Retourne : 
    - `float`: Le score métier normalisé, où 1 représente le meilleur score (aucun coût d'erreur), et 0 représente le pire score (coût maximal d'erreur).
    """
    
    #on sort les résultats de classification binaire
    tp, fp, fn, tn = confusion_matrix(y_test, y_pred).ravel()
    
    #on calcule les couts
    real_cost = fp * fp_coeff + fn * fn_coeff
    max_cost = (fp + fn) * (fp_coeff + fn_coeff)

    #on normalise
    normalized_cost = real_cost / max_cost

    #on inverse la tendance pour avoir que le score soit meilleur lorsque plus proche de 1
    score = 1 - normalized_cost

    return round(score, 3)



def mlflow_run_model(experiment_name, model, params, metric, n_folds, X_train, X_test, y_train, y_test, artifact_path, registered_model_name):
    
    """
    DOCUMENTATION :
    ---------------
    
    Entraîne un modèle de machine learning en utilisant GridSearchCV, logge les meilleurs paramètres et scores dans MLflow, 
    et enregistre le modèle dans le Model Registry de MLflow.

    Paramètres :
    ------------
    experiment_name : str
        Le nom de l'expérience MLflow dans laquelle les résultats seront enregistrés.

    model : scikit-learn estimator
        Le modèle de machine learning à entraîner (ex: RandomForestClassifier, LogisticRegression).

    params : dict
        Dictionnaire contenant la grille de recherche des hyperparamètres pour GridSearchCV.

    n_folds : int
        Le nombre de plis (folds) à utiliser pour la validation croisée dans GridSearchCV.

    X_train : array-like
        Les données d'entraînement utilisées pour ajuster le modèle.

    X_test : array-like
        Les données de test utilisées pour évaluer le modèle ajusté.

    y_train : array-like
        Les étiquettes des données d'entraînement.

    y_test : array-like
        Les étiquettes des données de test.

    artifact_path : str
        Le chemin sous lequel les artefacts du modèle seront enregistrés dans MLflow.

    registered_model_name : str
        Le nom sous lequel le modèle sera enregistré dans le Model Registry de MLflow.

    Retour :
    --------
    None
        Cette fonction ne retourne rien, mais elle logge les informations dans MLflow et imprime les meilleurs paramètres et scores.

    Étapes :
    --------
    1. Initialisation de l'expérience MLflow avec le nom donné.
    2. Entraînement du modèle avec GridSearchCV en utilisant la grille de paramètres spécifiée.
    3. Calcul des scores de validation croisée pour l'accuracy et le ROC AUC.
    4. Prédiction sur les données de test et calcul des métriques d'accuracy et de ROC AUC.
    5. Log des meilleurs paramètres, des scores de validation croisée et des scores de test dans MLflow.
    6. Enregistrement du modèle dans MLflow et enregistrement dans le Model Registry avec le nom spécifié.
    7. Affichage des meilleurs paramètres et scores dans la console.

    Exemple d'utilisation :
    -----------------------
    mlflow_run_model(
        experiment_name="Test Experiment",
        model=RandomForestClassifier(),
        params={'n_estimators': [100, 200], 'max_depth': [10, 20]},
        n_folds=5,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifact_path="random-forest",
        registered_model_name="RandomForestClassifierModel"
    )
    """

    #on initialise l'experimentation
    mlflow.set_experiment(experiment_name)

    #on définit la grille de recherche
    model_cv = GridSearchCV(estimator = model, param_grid = params, cv = n_folds , n_jobs = -1, verbose = 1, scoring = metric)

    with mlflow.start_run():
        model_cv.fit(X_train, y_train)

        #on sort les meilleurs paramètres et le meilleur candidat
        best_params = model_cv.best_params_
        best_model = model_cv.best_estimator_

        #on signe
        signature = infer_signature(X_train, best_model.predict(X_train))

        #on calcule les scores de validation croisée
        cv_accuracy = round(cross_val_score(best_model, X_train, y_train, cv = n_folds, scoring="accuracy", n_jobs = -1).mean(), 3)
        cv_rocauc = round(cross_val_score(best_model, X_train, y_train, cv = n_folds, scoring="roc_auc", n_jobs = -1).mean(), 3)
        cv_recall = round(cross_val_score(best_model, X_train, y_train, cv = n_folds, scoring="recall_macro", n_jobs = -1).mean(), 3)

        #on prédit
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:,1]

        #on sort les scores
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        rocauc = round(roc_auc_score(y_test, y_pred_proba), 3)
        recall = round(recall_score(y_test, y_pred), 3)
        f1 = round(f1_score(y_test, y_pred), 3)
        score_m = score_metier(y_test, y_pred, 1, 10)

        #on log les parametres et les scores
        mlflow.log_params(best_params)
        mlflow.log_metric("CV Accuracy", cv_accuracy)
        mlflow.log_metric("CV ROC AUC", cv_rocauc)
        mlflow.log_metric("CV Recall", cv_recall)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("Recall Score", recall)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("ROC AUC", rocauc)
        mlflow.log_metric("Score métier", score_m)

        #on log et on enregistre le modele
        mlflow.sklearn.log_model(
        sk_model = best_model,
        artifact_path = "sklearn" + "-" + artifact_path,
        signature = signature,
        input_example = X_train[:1],
        registered_model_name = "sklearn" + "-" + registered_model_name.lower(),
    )

    #on affiche tout de meme les résultats
    df_params = pd.DataFrame({"Paramètres": best_params.keys(), "Valeurs": best_params.values()})
    df_scores = pd.DataFrame({"Scores": ["CV Accuracy", "CV ROC AUC", "Accuracy", "ROC AUC", "SCORE METIER"], 
                              "Valeurs" : [cv_accuracy, cv_rocauc, accuracy, rocauc, score_m]})

    return df_params, df_scores


def feature_importance(model, X_train, X_test, columns_list, is_linear = True):

    """
    Cette fonction calcule et visualise l'importance des caractéristiques (features) d'un modèle de machine learning, en utilisant l'approche de valeurs de SHAP (SHapley Additive exPlanations). Elle peut gérer à la fois les modèles linéaires et non linéaires et offre différentes visualisations interactives pour explorer l'importance globale et locale des caractéristiques.

    Paramètres
    ----------
    model : object
        Le modèle de machine learning entraîné (par exemple, régression logistique, arbre de décision, etc.) 
        qui possède un attribut `coef_` pour les modèles linéaires ou `feature_importances_` pour les modèles non linéaires.
    
    X_train : numpy.ndarray
        Le tableau de données d'entraînement utilisé pour ajuster le modèle.
        Les données doivent être structurées de manière cohérente avec `X_test` et les colonnes doivent être alignées avec `columns_list`.
    
    X_test : numpy.ndarray
        Le tableau de données de test sur lequel l'importance des caractéristiques sera calculée.
        Les données doivent être structurées de manière cohérente avec `X_train`.
    
    columns_list : list of str
        Une liste de chaînes de caractères représentant les noms des colonnes (features) utilisées dans le modèle.
    
    is_linear : bool, optional (default=True)
        Un booléen indiquant si le modèle est linéaire (`True`) ou non linéaire (`False`). 
        Si `True`, la fonction utilisera les coefficients du modèle linéaire (`coef_`). Si `False`, 
        elle utilisera l'attribut `feature_importances_` pour les modèles non linéaires.

    Returns
    -------
    data_features_importance : pandas.DataFrame
        Un DataFrame contenant les caractéristiques (features) avec leurs importances associées, normalisées et exprimées en pourcentage.
    
    data_shap : pandas.DataFrame
        Un DataFrame contenant les valeurs de SHAP pour chaque observation dans `X_test`.
        Chaque colonne correspond à une caractéristique (feature) et chaque ligne à une observation.

    Notes
    -----
    * La fonction interroge l'utilisateur pour afficher différents graphiques :
      - Graphique d'importance globale des caractéristiques (bar plot).
      - Diagramme en barres des valeurs de SHAP.
      - Diagramme en essaim (beeswarm) des valeurs de SHAP.
      - Graphique waterfall pour expliquer l'importance locale des caractéristiques pour une observation spécifique.
    
    * Les visualisations des graphiques dépendent de la bibliothèque `matplotlib` et `shap` pour le tracé des valeurs de SHAP.
    * La fonction inclut également des vérifications des erreurs d'entrée pour éviter les erreurs d'exécution.

    Exemple
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    
    >>> # Chargement du dataset
    >>> data = load_iris()
    >>> X = np.array(data.data)
    >>> y = data.target

    >>> # Division des données en train et test
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    >>> # Entraînement d'un modèle linéaire
    >>> model = LogisticRegression(max_iter=200)
    >>> model.fit(X_train, y_train)
    
    >>> # Utilisation de la fonction pour visualiser l'importance des caractéristiques
    >>> columns_list = data.feature_names
    >>> data_features_importance, data_shap = feature_importance(model, X_train, X_test, columns_list, is_linear = True) 
    """

    #syntaxe latex pose problème avec les '_' donc on la désactive pour être que que tout fonctionne
    plt.rc('text', usetex = False)
    
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Latin Modern Math', 'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    
    if X_train.shape[1] != len(columns_list):
        print("Erreur : Le nombre de colonnes de X_train doit correspondre à la longueur de la liste de colonnes")
        return None

    #on initialise un explainer shap
    explainer = shap.Explainer(model, X_train)
    explainer.expected_value

    
    data_shap = pd.DataFrame()

    if is_linear == True:
        #on sort les coefficients du modele linéaire
        feature_importance = model.coef_[0]

        #les shap values
        shap_values = explainer(X_test)
    else:
        #on sort la feature importance
        feature_importance = model.feature_importances_.tolist()

        #les shap values
        shap_values = explainer(X_test, check_additivity = False)
    

    #on cree un dataframe avec les features importances
    data_features_importance = pd.DataFrame({'Feature' : columns_list, 'Importance' : feature_importance})
    data_features_importance['Absolute Importance'] = data_features_importance['Importance'].apply(np.abs)

    sum_importance = data_features_importance['Absolute Importance'].sum()

    data_features_importance['Normalized Importance'] = data_features_importance['Absolute Importance'] / sum_importance
    data_features_importance['Importance Percentage'] = data_features_importance['Normalized Importance'] * 100

    data_features_importance = data_features_importance.sort_values(by = 'Importance Percentage', ascending = False)

    #on cree le dataframe avec les shap values
    data_shap = pd.DataFrame(shap_values.values, columns = columns_list)

    print('*' * 80)
    print('-' * 21 + ' Bienvenue dans la FEATURE IMPORTANCE ' + '-' * 21)
    print('_'* 80)
    print('-' * 80)
    
    #demande a l'utilisateur s'il veut afficher le graphique feature importance
    show_feature_importance = input("Voulez-vous afficher le graphique de l'importance des caractéristiques ? (y/n) : ")
    if show_feature_importance == 'y':
        print('-' * 80)
        n = input("Entrez le nombre de caractéristiques souhaitées (max 40) : ")
        print('-' * 80)

        # Vérifier que la valeur de n saisie soit un entier
        try:
            n = int(n)
            if n > 0 and n <= 40:
                # Plot du dataframe de feature_importances
                plt.figure(figsize=(12, round(n / 2) + 1))
                sns.barplot(
                    x=data_features_importance['Importance Percentage'].head(n), 
                    y=data_features_importance['Feature'].head(n), 
                    palette='dark:blue',
                    hue=data_features_importance['Feature'].head(n),
                    dodge=False
                )
                plt.xlabel('Importance (%)', size=14)
                plt.xticks(size=14)
                plt.yticks(size=14)
                plt.ylabel('')
                plt.title(f"Importance des caractéristiques - Top {n}", size=20, pad=20, fontweight='bold')
                plt.show()
                plt.close()
                print('-' * 80)
            else:
                print("Erreur : Veuillez entrer un nombre entre 1 et 40.")

        except ValueError:
            print("Erreur : Vous devez entrer un nombre entier valide.")

    #demande a l'utilisateur s'il veut afficher le plotsbar de shap
    show_shap_pots_bar = input("Voulez vous afficher le diagramme en barre de SHAP ? (y/n) : ")

    if show_shap_pots_bar == 'y':
        print('-' * 80)
        #on plot le plotsbar de shap
        shap_values.feature_names = columns_list
        shap.plots.bar(shap_values)
        plt.close()
        print('-' * 80)

    #demande a l'utilisateur s'il veut afficher le diagramme en essaim shap
    show_shap_beeswarm = input("Voulez vous afficher le diagramme en essaim de SHAP ? (y/n) : ")

    if show_shap_beeswarm == 'y':
        print('-' * 80)
        #on plot le diagramme en abeille
        shap.plots.beeswarm(shap_values, max_display = 10, plot_size = (12,6))
        plt.close()
        print('-' * 80)
    
    #demande a l'utilisateur s'il veut afficher la feature importance locale
    show_local_explanation = input("Voulez-vous afficher l'importance des caractéristiques locale ? (y/n) : ")
    print('-'*80)

    if show_local_explanation == 'y':
        i = input("Entrez l'index de l'observation pour laquelle vous souhaitez visualiser l'importance des caractéristiques : ")
        print('-' * 80)

        #on vérifie que i soit entier et inférieur au nombre de lignes du jeu de données
        try:
            i = int(i)
            # Vérifier que i soit entier et inférieur au nombre de lignes du jeu de données
            if 0 <= i < X_train.shape[0]:
                # Plot de l'explication locale avec waterfall plot
                explanation_i = shap.Explanation(
                    values=shap_values[i], 
                    base_values=shap_values.base_values, 
                    data=X_train[i,:], 
                    feature_names=columns_list
                )
                shap.waterfall_plot(explanation_i, show=False)
                fig = plt.gcf()
                fig_width, fig_height = fig.get_size_inches()
                fig.set_size_inches(fig_width * 0.8, fig_height * 1)
                plt.show()
                plt.close()
                print('-' * 80)
            else:
                print("Erreur : L'index doit être un entier compris entre 0 et le nombre total d'observations.")

        except ValueError:
            print("Erreur : Vous devez entrer un nombre entier valide.")

    return data_features_importance, data_shap


def threshold_optimization(y_test, y_proba, thresholds_range):
    
    """
    Optimise le seuil de classification d'un modèle de machine learning basé sur plusieurs métriques de performance et renvoie le seuil optimal.

    Paramètres:
        y_test (array-like ou liste): Vecteur des vraies étiquettes (valeurs cibles) pour les données de test.
        y_proba (array-like): Probabilités prédites par le modèle pour les données de test.
        thresholds_range (array-like): Liste des seuils à tester pour déterminer le seuil optimal de classification.

    Returns:
        float: Le seuil de classification optimal qui maximise le score général.

    Description:
        La fonction `threshold_optimization` calcule plusieurs métriques de performance (score métier, accuracy, et ROC AUC) pour différents seuils de classification. 
        Elle détermine ensuite le seuil optimal qui maximise une combinaison pondérée de ces métriques.

        - Pour chaque seuil dans `thresholds_range`, la fonction calcule les prédictions binaires (`y_pred`) en comparant les probabilités prédites (`y_proba`) au seuil.
        - Les métriques suivantes sont calculées pour chaque seuil :
            - Score Métier : Calculé à l'aide d'une fonction de score métier personnalisée (`fc.score_metier`) avec des coûts de faux positifs et de faux négatifs pondérés.
            - Accuracy : Proportion de prédictions correctes.
            - ROC AUC : Aire sous la courbe ROC, une mesure de la capacité du modèle à distinguer entre les classes.
        - Un score général est calculé pour chaque seuil en combinant les trois métriques de performance avec les pondérations suivantes : 0.5 pour le score métier, 0.2 pour l'accuracy, et 0.3 pour le ROC AUC.
        - Les scores sont triés par ordre décroissant de score général. Le seuil optimal est celui qui a le score général le plus élevé.

    Example:
        >>> best_threshold = threshold_optimization(y_test, y_pred_proba, thresholds_range)
        ------------------------------------------------------------------------------
        Seuil optimal : 0.3
        ---------------------
        ROC AUC Score : 0.85
        Accuracy Score : 0.80
        Score Métier : 0.75
        ------------------------------------------------------------------------------
        Voulez-vous afficher le graphique des scores en fonction des seuils ? (y/n) : y
        [Graphique affiché]

    Notes:
        - Assurez-vous que les bibliothèques nécessaires (Pandas, Matplotlib, Seaborn) sont importées et que la fonction `fc.score_metier` est définie dans votre environnement.
        - Les pondérations pour le calcul du score général (`0.5`, `0.2`, `0.3`) peuvent être ajustées selon les priorités du problème.

    """
    
    scores_metier = []
    scores_accuracy = []
    scores_sum = []
    
    for threshold in thresholds_range:
        y_pred = (y_proba >= threshold).astype(int)
        metier = score_metier(y_test, y_pred, 1, 10)
        accuracy = accuracy_score(y_test, y_pred)

        score_sum = 0.75 * metier + 0.25 * accuracy
        
        scores_metier.append(metier)
        scores_accuracy.append(accuracy)
        scores_sum.append(score_sum)
        
    data_scores = pd.DataFrame({
        'threshold': thresholds_range.tolist(),
        'performance_score': scores_metier,
        'accuracy_score': scores_accuracy,
        'general_score': scores_sum
    })

    data_scores_sorted = data_scores.sort_values(by = 'general_score', ascending = False)
    best_threshold = data_scores_sorted['threshold'].values[0]

    best_acc = data_scores_sorted['accuracy_score'].values[0]
    best_metier = data_scores_sorted['performance_score'].values[0]

    print('-' * 80)
    print(f"Seuil optimal : {best_threshold}")
    print('-' * 21)
    print(f"Accuracy Score : {round(best_acc, 2)}")
    print(f"Score Métier : {round(best_metier, 2)}")
    print('-' * 80)
    
    plot = input("Voulez-vous afficher le graphique des scores en fonction des seuils ? (y/n) : ")
    
    if plot == 'y':
        print('-' * 80)
        plt.figure(figsize=(14,8))
        sns.lineplot(x = data_scores['threshold'], y = data_scores['performance_score'], color = 'blue', label = 'Score métier')
        sns.lineplot(x = data_scores['threshold'], y = data_scores['accuracy_score'], color = 'green', label = 'Accuracy')
        plt.title(f"Scores en fonction du seuil", fontweight = 'bold', size = 20, pad = 20)
        plt.axvline(x = best_threshold, color = 'r', linestyle = '--', label = 'Seuil optimal')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()
        plt.show()
        
    return best_threshold