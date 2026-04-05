import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=tuple(self.config.params_image_size)
        )
        self.model.trainable = False
        self.save_model(self.config.base_model_path, self.model)

    @staticmethod
    def _add_head(base_model, num_classes, dense_1, dense_2, dropout_1, dropout_2):
        base_model.trainable = False
        inputs = layers.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(dense_1, activation="relu")(x)
        x = layers.Dropout(dropout_1)(x)
        x = layers.Dense(dense_2, activation="relu")(x)
        x = layers.Dropout(dropout_2)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return Model(inputs, outputs)

    def update_base_model(self):
        self.full_model = self._add_head(
            base_model=self.model,
            num_classes=self.config.params_num_classes,
            dense_1=self.config.params_dense_1,
            dense_2=self.config.params_dense_2,
            dropout_1=self.config.params_dropout_1,
            dropout_2=self.config.params_dropout_2
        )
        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)