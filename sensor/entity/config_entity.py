import os, sys
from sensor.exception import SensorException
from sensor.logger import logging
from datetime import datetime

FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJ_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJ_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:
    # whenever we are running this we are creating a new folder each time with timestamp

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise SensorException(e,sys)


class DataIngestionConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = "aps"
            self.collection_name = "sensor"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise SensorException(e,sys)

    def to_dict(self,)->dict:
        # convert every thing passed into dictionary format
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e,sys)

class DataValidationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        # create some validation report(info about drift, anomaly, etc) file within above created folder
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml") 
        self.missing_threshold:float = 0.2
        self.base_file_path = os.path.join("aps_failure_training_set1.csv")
        
class DataTransformationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        # to create a data_transformation folder in the timestamp folder inside the artifact directory
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        # storing data transformation object for future use in prediction pipelines
        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_OBJ_FILE_NAME)
        self.transformed_train_path = os.path.join(self.data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transformed_test_path = os.path.join(self.data_transformation_dir, "transformed", TEST_FILE_NAME.replace("csv", "npz"))
        # file for target encoding
        self.target_encoder_path = os.path.join(self.data_transformation_dir, "target_encoder", TARGET_ENCODER_OBJ_FILE_NAME)

class ModelTrainerConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        # save our model in model_path
        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.1

class ModelEvaluationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        # if train model is performing better by 1%, then we can accept it
        self.change_threshold = 0.01

class ModelPusherConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        # creating a directory named saved_models outside artifact directory i.e., in root directory
        self.saved_models_dir = os.path.join("saved_models")
        # also creating a directory with name saved_models inside artifact directory
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        # above file path is artifact/model_pusher/saved_models
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir, TRANSFORMER_OBJ_FILE_NAME)
        self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir, TARGET_ENCODER_OBJ_FILE_NAME)