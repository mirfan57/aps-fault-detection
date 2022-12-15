import os, sys
import numpy as np
import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score



class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        # we will train our model on the output of data transformation stage
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self):
        try:
            # Write code for GridSearchCV
            pass

        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x, y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test array")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info("Train the model after splitting feature and target from array")
            model = self.train_model(x=x_train, y=y_train)

            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_train, yhat_train)

            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_test, yhat_test)

            logging.info(f"train score:{f1_train_score} and test score {f1_test_score}")

            # check for overfitting, underfitting or expected score(defined by user)
            # dealing underfitting
            logging.info("Checking whether model is underfitting or not by comparing with expected accuracy")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is underperforming as obtained accuracy: {f1_test_score} is lower than expected: \
                               {self.model_trainer_config.expected_score}")

            logging.info(f"Checking whether model is overfitting or not")
            diff = abs(f1_train_score-f1_test_score)
            
            # dealing overfitting
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score difference: {diff} is more than set overfitting \
                                threshold: {self.model_trainer_config.overfitting_threshold}")

            # save the trained model
            logging.info("Saving trained model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
                                                f1_train_score = f1_train_score, f1_test_score = f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
