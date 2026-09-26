import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    """
    Fonction principale permettant d'exécuter le pipeline ML.
    """

    parser = argparse.ArgumentParser(
        description="Pipeline de classification Customer Churn"
    )

    parser.add_argument(
        "--data", default="Churn_Modelling (1).csv", help="Chemin vers le fichier CSV"
    )

    parser.add_argument(
        "--model", default="classifier.joblib", help="Chemin du modèle sauvegardé"
    )

    args = parser.parse_args()

    # 1. Préparation des données
    print("\n=== 1. Préparation des données ===")
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    print(f"Nombre de données d'entraînement : {len(X_train)}")
    print(f"Nombre de données de test : {len(X_test)}")

    # 2. Entraînement du modèle
    print("\n=== 2. Entraînement du modèle ===")
    model = train_model(X_train, y_train)

    print("Random Forest entraîné avec succès.")

    # 3. Évaluation du modèle
    print("\n=== 3. Évaluation du modèle ===")
    accuracy, matrix = evaluate_model(model, X_test, y_test)

    # 4. Sauvegarde du modèle
    print("\n=== 4. Sauvegarde du modèle ===")
    save_model(model, args.model)

    # 5. Chargement du modèle
    print("\n=== 5. Chargement du modèle ===")
    loaded_model = load_model(args.model)

    print("Pipeline terminé avec succès.")


if __name__ == "__main__":
    main()
