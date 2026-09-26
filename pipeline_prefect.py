import argparse
import subprocess
import os
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# ============================================================
# TASKS
# ============================================================


@task
def install_dependencies():
    """Installe les dépendances du projet."""
    print("\n=== Installation des dépendances ===")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
    )

    print("Dépendances installées avec succès.")


@task
def prepare_data_task(data_filename="Churn_Modelling (1).csv"):
    """Prépare les données."""
    print("\n=== Préparation des données ===")

    X_train, X_test, y_train, y_test = prepare_data(data_filename)

    print(f"Nombre de données d'entraînement : {len(X_train)}")
    print(f"Nombre de données de test : {len(X_test)}")

    return X_train, X_test, y_train, y_test


@task
def train_model_task(X_train, y_train):
    """Entraîne le modèle."""
    print("\n=== Entraînement du modèle ===")

    model = train_model(X_train, y_train)

    print("Random Forest entraîné avec succès.")

    return model


@task
def save_model_task(model, model_filename="classifier.joblib"):
    """Sauvegarde le modèle."""
    print("\n=== Sauvegarde du modèle ===")

    save_model(model, model_filename)

    return model_filename


@task
def load_model_task(model_filename="classifier.joblib"):
    """Charge le modèle sauvegardé."""
    print("\n=== Chargement du modèle ===")

    model = load_model(model_filename)

    return model


@task
def evaluate_model_task(model, X_test, y_test):
    """Évalue le modèle."""
    print("\n=== Évaluation du modèle ===")

    accuracy, matrix = evaluate_model(model, X_test, y_test)

    return accuracy, matrix


def format_code():
    """Formate les fichiers Python du projet avec Black."""
    print("\n=== Formatage du code ===")

    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]

    subprocess.run([sys.executable, "-m", "black", *files], check=True)

    print("Formatage terminé.")


@task
def check_code_quality():
    """Analyse la qualité du code avec Pylint."""
    print("\n=== Vérification de la qualité du code ===")

    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]

    subprocess.run([sys.executable, "-m", "pylint", *files], check=False)

    print("Analyse de qualité terminée.")


@task
def check_code_security():
    """Analyse la sécurité du code avec Bandit."""
    print("\n=== Vérification de la sécurité ===")

    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]

    subprocess.run([sys.executable, "-m", "bandit", "-r", *files], check=False)

    print("Analyse de sécurité terminée.")


@task
def run_tests():
    """Exécute les tests unitaires."""
    print("\n=== Exécution des tests unitaires ===")

    subprocess.run([sys.executable, "-m", "pytest"], check=True)

    print("Tests unitaires réussis.")


@task
def get_project():
    """Récupère le projet depuis GitHub."""
    print("\n=== Récupération du projet depuis GitHub ===")

    repo_url = "https://github.com/Yasmine-Benmanaa/Atelier3_MLOPS.git"
    project_dir = "../ml_project_remote"

    if not os.path.exists(project_dir):
        print("Le projet n'existe pas localement. Clonage...")
        subprocess.run(["git", "clone", repo_url, project_dir], check=True)
    else:
        print("Le projet existe déjà. Mise à jour...")
        subprocess.run(["git", "-C", project_dir, "pull"], check=True)

    print("Projet récupéré avec succès.")


# ============================================================
# FLOWS
# ============================================================


@flow(name="ml-pipeline-all")
def all_flow():
    """Exécute le pipeline complet."""
    # . Récupération du projet depuis GitHub
    get_project()

    # 1. Vérification du code
    code_flow()

    # 2. Préparation des données
    X_train, X_test, y_train, y_test = prepare_data_task()

    # 3. Entraînement
    model = train_model_task(X_train, y_train)

    # 4. Sauvegarde
    save_model_task(model)

    # 5. Évaluation
    evaluate_model_task(model, X_test, y_test)


@flow(name="ml-pipeline-train")
def train_flow():
    """Exécute la préparation et l'entraînement."""

    X_train, X_test, y_train, y_test = prepare_data_task()

    train_model_task(X_train, y_train)


@flow(name="ml-pipeline-evaluate")
def evaluate_flow():
    """Charge et évalue le modèle."""

    X_train, X_test, y_train, y_test = prepare_data_task()

    model = load_model_task()

    evaluate_model_task(model, X_test, y_test)


@flow(name="ml-pipeline-code")
def code_flow():
    """Exécute les vérifications du code."""

    install_dependencies()
    format_code()
    check_code_quality()
    check_code_security()
    run_tests()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")

    parser.add_argument(
        "--flow",
        choices=["all", "code", "train", "entrainement", "evaluate"],
        required=True,
        help="Flow à exécuter",
    )

    args = parser.parse_args()

    if args.flow == "code":
        code_flow()

    elif args.flow == "all":
        all_flow()

    elif args.flow in ["train", "entrainement"]:
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()
