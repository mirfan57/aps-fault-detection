import os, sys
import pandas as pd
import numpy as np
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from sklearn.pipeline import Pipeline
from sensor import utils
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer     # populate some values for the missing rows
from sklearn.preprocessing import RobustScaler    # scaling to minimize the effect of outliers
from imblearn.combine import SMOTETomek    # to generate some data for the minority class
from sensor.config import TARGET_COLUMN


class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig, 
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e, sys)
    
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            # If strategy="constant", then replace missing values with fill_value
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()
            # creating pipeline in the order we want to execute it
            pipeline = Pipeline(steps = [('Imputer',simple_imputer), ('RobustScaler', robust_scaler)])  
            return pipeline
        except Exception as e:
            raise SensorException(e, sys) 
        

    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(columns=TARGET_COLUMN)
            input_feature_test_df = test_df.drop(columns=TARGET_COLUMN)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # fitting label encoder on top of target col of train df
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            # since our target column is categorical, we want to convert it into numerical: "pos" to 1 and "neg" to 0
            # transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # transforming input features using pipeline having simple imputer, robust scaler
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            # using library to balance our dataset as "neg" class is around 35K and "pos" class are only 1K
            smt = SMOTETomek(random_state=21)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr.shape}")


            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr.shape}")

            # concatenate input and target feature arrays using numpy.c_ so that we can save it as a single array 
            # numpy.c_ translates slice objects to concatenation along the second axis.

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            
            # save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            # saving transformation pipeline for future use
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            # saving target encoder
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                                           transform_object_path=self.data_transformation_config.transform_object_path, 
                                           transformed_train_path=self.data_transformation_config.transformed_train_path, 
                                           transformed_test_path=self.data_transformation_config.transformed_test_path, 
                                           target_encoder_path=self.data_transformation_config.target_encoder_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

