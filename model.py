"""Regularizowany model 3D-CNN do rozpoznawania czynności w wideo."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


L2_WEIGHT_DECAY = 1e-4


@tf.keras.utils.register_keras_serializable(package="action_recognition")
class VideoAugmentation(layers.Layer):
    """Augmentuje cały klip wyłącznie podczas treningu."""

    def __init__(
        self,
        brightness_delta=0.08,
        contrast_min=0.85,
        contrast_max=1.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brightness_delta = brightness_delta
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max

    def call(self, inputs, training=None):
        def augment():
            # TensorFlow image helpers accept only 3D/4D tensors.  Here a
            # sample is a 5D video: batch × time × height × width × channels.
            # One transform is drawn per video, then applied to every frame.
            batch_size = tf.shape(inputs)[0]
            shape = (batch_size, 1, 1, 1, 1)

            flip_mask = tf.random.uniform(shape) < 0.5
            videos = tf.where(flip_mask, tf.reverse(inputs, axis=[3]), inputs)

            brightness = tf.random.uniform(
                shape, -self.brightness_delta, self.brightness_delta
            )
            videos = videos + brightness

            contrast = tf.random.uniform(
                shape, self.contrast_min, self.contrast_max
            )
            frame_mean = tf.reduce_mean(videos, axis=[2, 3], keepdims=True)
            videos = (videos - frame_mean) * contrast + frame_mean
            return tf.clip_by_value(videos, 0.0, 1.0)

        if training is None or training is False:
            return inputs
        if isinstance(training, bool):
            return augment()
        return tf.cond(training, augment, lambda: inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "brightness_delta": self.brightness_delta,
                "contrast_min": self.contrast_min,
                "contrast_max": self.contrast_max,
            }
        )
        return config


def create_model(input_shape, num_classes):
    """Tworzy sprawdzony, regularizowany model 3D-CNN."""
    regularizer = regularizers.l2(L2_WEIGHT_DECAY)
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            VideoAugmentation(name="video_augmentation"),
            layers.Conv3D(
                64, (3, 3, 3), use_bias=False, kernel_regularizer=regularizer
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling3D((2, 2, 2)),
            layers.SpatialDropout3D(0.10),
            layers.Conv3D(
                128, (3, 3, 3), use_bias=False, kernel_regularizer=regularizer
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling3D((2, 2, 2)),
            layers.SpatialDropout3D(0.15),
            layers.Conv3D(
                256, (6, 1, 1), use_bias=False, kernel_regularizer=regularizer
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv3D(
                512, (1, 1, 1), use_bias=False, kernel_regularizer=regularizer
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling3D((1, 1, 1)),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizer),
            layers.Flatten(),
            layers.Dropout(0.45),
            layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
