import numpy as np
import tensorflow as tf


class PredictionPipeline:
    def __init__(self, model_path: str):
        self.model     = tf.saved_model.load(model_path)  # ← SavedModel loader
        self.infer     = self.model.signatures["serving_default"]
        self.classes   = ["Cyst", "Tumor", "Stone", "Normal"]
        self.img_size  = (224, 224)

    def predict(self, image_path: str) -> dict:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, axis=0)

        output     = self.infer(img)
        probs      = list(output.values())[0].numpy()[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(np.max(probs)) * 100

        return {
            "prediction": pred_class,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {
                cls: f"{float(p)*100:.2f}%"
                for cls, p in zip(self.classes, probs)
            }
        }