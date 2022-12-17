import os, sys
import pandas as pd
from sensor.entity import config_entity, artifact_entity
from sensor.predictor import ModelResolver
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_object
from sensor.config import TARGET_COLUMN
from sklearn.metrics import f1_score

class ModelEvaluation:

    def __init__(self, model_eval_config:config_entity.ModelEvaluationConfig, 
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact, 
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved_models folder alrady has model the we will compare 
            # which model is the best: currently trained one or the model from saved model folder
            logging.info(f"If model present in saved_models folder then we will compare which model is best: recently trained \
            model or the latest model in saved_models folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of transformer model and target encoder
            logging.info(f"Finding location of currently accepted model, transformer and target encoder")
            model_path = self.model_resolver.get_latest_model_path()
            transformer_path = self.model_resolver.get_latest_transformer_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info(f"Previously trained objects of model, transformer and target encoder")
            # Previously trained objects
            model = load_object(file_path=model_path)
            transformer = load_object(file_path=transformer_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # Finding location of transformer model and target encoder
            logging.info(f"Finding location of currently trained model, transformer and target encoder")
            current_model_path = self.model_trainer_artifact.model_path
            current_transformer_path = self.data_transformation_artifact.transform_object_path
            current_target_encoder_path = self.data_transformation_artifact.target_encoder_path

            logging.info(f"Currently trained objects of model, transformer and target encoder")
            # Currently trained model objects
            current_model = load_object(file_path=current_model_path)
            current_transformer = load_object(file_path=current_transformer_path)
            current_target_encoder = load_object(file_path=current_target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)

            # accuracy of previously trained and currently utilised model
            # as aour transformer obj may have different feature cols as compared to test_df, we extract the input features first
            input_feature_name = list(transformer.feature_names_in_)        # to get all feature cols present after removing cols having different distributions of that particular train_df
            input_arr = transformer.transform(test_df[input_feature_name])    # passing only those features for transformation
            y_pred = model.predict(input_arr)
            logging.info(f"Fetching first 5 predictions from currently accepted model")
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")

            # Accuracy using currently trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            # transforming input feature columnms
            input_arr = current_transformer.transform(test_df[input_feature_name])
            y_pred_current = current_model.predict(input_arr)
            y_true_current = current_target_encoder.transform(target_df)
            logging.info(f"Fetching first 5 predictions from currently trained model")
            print(f"Prediction using currently trained model: {current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true = y_true_current, y_pred = y_pred_current)
            logging.info(f"Accuracy using currently trained model: {current_model_score}")

            # Comparing current and previous model scores
            if current_model_score <= previous_model_score:
                logging.info(f"Our current trained model is not better than currently accepted model")
                raise Exception("Our current trained model is not better than currently accepted model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact


        except Exception as e:
            raise SensorException(e, sys)