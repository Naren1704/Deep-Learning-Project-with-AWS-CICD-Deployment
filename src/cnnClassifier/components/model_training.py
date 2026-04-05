import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def _load_data(self):
        classes     = self.config.params_classes
        cls_to_idx  = {cls: i for i, cls in enumerate(classes)}
        all_paths, all_labels = [], []

        for cls in classes:
            folder = os.path.join(self.config.training_data, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_paths.append(os.path.join(folder, fname))
                    all_labels.append(cls_to_idx[cls])

        all_paths  = np.array(all_paths)
        all_labels = np.array(all_labels)

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            all_paths, all_labels, test_size=0.30,
            stratify=all_labels, random_state=self.config.params_seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50,
            stratify=y_tmp, random_state=self.config.params_seed
        )
        logger.info(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_tr, X_val, X_test, y_tr, y_val, y_test

    def _preprocess(self, path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.config.params_image_size[:2])
        img = tf.cast(img, tf.float32)   # EfficientNetB0 handles its own scaling
        return img, label

    def _augment(self, img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1 * 255)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = tf.image.rot90(img, k=tf.random.uniform(
            shape=[], minval=0, maxval=4, dtype=tf.int32))
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    def _build_dataset(self, paths, labels, augment=False, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=self.config.params_seed)
        ds = ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)

    def train(self):
        X_tr, X_val, X_test, y_tr, y_val, y_test = self._load_data()

        train_ds = self._build_dataset(X_tr,  y_tr,  augment=True,  shuffle=True)
        val_ds   = self._build_dataset(X_val, y_val, augment=False, shuffle=False)

        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # ── Phase 1: frozen base ───────────────────────────────────────────────
        logger.info("Phase 1: Training head only")
        self.model.compile(
            optimizer=Adam(self.config.params_lr_phase1),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        ckpt_dir = os.path.join(self.config.root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        cb1 = [
            ModelCheckpoint(os.path.join(ckpt_dir, "best_phase1.keras"),
                            monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=5,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)
        ]

        self.model.fit(
            train_ds, epochs=self.config.params_epochs_phase1,
            validation_data=val_ds,
            class_weight=self.config.params_class_weights,
            callbacks=cb1
        )

        # ── Phase 2: fine-tune top layers ──────────────────────────────────────
        logger.info("Phase 2: Fine-tuning top layers")
        # Unfreeze top FINE_TUNE_LAYERS layers of EfficientNetB0
        base = self.model.layers[1]   # EfficientNetB0 is layer index 1
        base.trainable = True
        for layer in base.layers[:-self.config.params_fine_tune_layers]:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(self.config.params_lr_phase2),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        cb2 = [
            ModelCheckpoint(os.path.join(ckpt_dir, "best_phase2.keras"),
                            monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=6,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-7, verbose=1)
        ]

        self.model.fit(
            train_ds, epochs=self.config.params_epochs_phase2,
            validation_data=val_ds,
            class_weight=self.config.params_class_weights,
            callbacks=cb2
        )

        self.model.save(self.config.trained_model_path)
        logger.info(f"Model saved at {self.config.trained_model_path}")