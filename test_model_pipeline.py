from model_pipeline import train_model


def test_train_model():
    X_train = [
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
    ]

    y_train = [0, 0, 1, 1]

    model = train_model(X_train, y_train)

    assert model is not None
