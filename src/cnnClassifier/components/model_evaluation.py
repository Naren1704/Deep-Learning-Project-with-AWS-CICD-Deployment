import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger
import os


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _load_test_data(self):
        classes    = self.config.params_classes
        cls_to_idx = {cls: i for i, cls in enumerate(classes)}
        data_dir   = os.path.join(
            self.config.training_data,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )
        all_paths, all_labels = [], []
        for cls in classes:
            folder = os.path.join(data_dir, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_paths.append(os.path.join(folder, fname))
                    all_labels.append(cls_to_idx[cls])

        all_paths  = np.array(all_paths)
        all_labels = np.array(all_labels)

        _, X_tmp, _, y_tmp = train_test_split(
            all_paths, all_labels, test_size=0.30,
            stratify=all_labels, random_state=42
        )
        _, X_test, _, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50,
            stratify=y_tmp, random_state=42
        )
        return X_test, y_test

    def _preprocess(self, path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.config.params_image_size[:2])
        img = tf.cast(img, tf.float32)
        return img, label

    def evaluation(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        X_test, y_test = self._load_test_data()

        test_ds = (
            tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.params_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        loss, accuracy = self.model.evaluate(test_ds)
        y_pred = np.argmax(self.model.predict(test_ds), axis=1)
        report = classification_report(
            y_test, y_pred,
            target_names=self.config.params_classes,
            output_dict=True
        )

        self.scores = {
            "loss": loss,
            "accuracy": accuracy,
            "classification_report": report
        }
        save_json(Path("scores.json"), self.scores)
        logger.info(f"Test Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss":     self.scores["loss"],
                "accuracy": self.scores["accuracy"]
            })

            if tracking_url_type != "file":
                mlflow.keras.log_model(
                    self.model, "model",
                    registered_model_name="EfficientNetB0KidneyClassifier"
                )
            else:
                mlflow.keras.log_model(self.model, "model")