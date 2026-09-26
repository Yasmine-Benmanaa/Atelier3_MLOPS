import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def prepare_data(data_filename):
    """
    Charge et prétraite les données du projet Customer Churn.

    Parameters
    ----------
    data_filename : str
        Chemin vers le fichier CSV.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Données d'entraînement et de test.
    """

    # Chargement des données
    df = pd.read_csv(data_filename)

    # Encodage de la variable Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns=columns_to_drop)

    # Séparation des variables explicatives et de la cible
    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # Suppression des identifiants
    X = X.drop(columns=["RowNumber", "CustomerId"])

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîne un modèle Random Forest.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Données d'entraînement.
    y_train : pandas.Series
        Variable cible d'entraînement.

    Returns
    -------
    RandomForestClassifier
        Modèle entraîné.
    """

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle avec l'accuracy et la matrice de confusion.

    Parameters
    ----------
    model : modèle entraîné
        Modèle de classification.
    X_test : pandas.DataFrame
        Données de test.
    y_test : pandas.Series
        Variable cible de test.

    Returns
    -------
    accuracy : float
        Score d'accuracy.
    matrix : numpy.ndarray
        Matrice de confusion.
    """

    # Prédictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Matrice de confusion
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("\nMatrice de confusion :")
    print(matrix)

    return accuracy, matrix


def save_model(model, filename="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné avec Joblib.

    Parameters
    ----------
    model : modèle entraîné
        Modèle à sauvegarder.
    filename : str
        Nom du fichier de sauvegarde.
    """

    joblib.dump(model, filename)

    print(f"Modèle sauvegardé dans : {filename}")


def load_model(filename="classifier.joblib"):
    """
    Charge un modèle sauvegardé.

    Parameters
    ----------
    filename : str
        Chemin du modèle sauvegardé.

    Returns
    -------
    model
        Modèle chargé.
    """

    model = joblib.load(filename)

    print(f"Modèle chargé depuis : {filename}")

    return model
