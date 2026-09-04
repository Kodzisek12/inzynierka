from tensorflow.keras import layers, models


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(64, (3, 3, 3), activation="relu"),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        layers.Conv3D(128, (3, 3, 3), activation="relu"),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        layers.Conv3D(256, (6, 1, 1), activation="relu"),
        layers.Conv3D(512, (1, 1, 1), activation="relu"),
        layers.MaxPooling3D((1, 1, 1)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
