from cnnClassifier import logger
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

# Stage 1: Data already in Kaggle — skip ingestion if running locally
# Stage 2: Prepare base model
STAGE = "Prepare Base Model"
try:
    logger.info(f">>>>>> stage {STAGE} started <<<<<<")
    PrepareBaseModelTrainingPipeline().main()
    logger.info(f">>>>>> stage {STAGE} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e); raise e

# Stage 3: Training
STAGE = "Model Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE} started <<<<<<")
    ModelTrainingPipeline().main()
    logger.info(f">>>>>> stage {STAGE} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e); raise e

# Stage 4: Evaluation
STAGE = "Model Evaluation"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE} started <<<<<<")
    EvaluationPipeline().main()
    logger.info(f">>>>>> stage {STAGE} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e); raise e