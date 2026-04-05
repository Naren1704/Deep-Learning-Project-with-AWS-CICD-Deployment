import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)
logger = logging.getLogger("cnnClassifier")