import os, sys
import pandas as pd
import numpy as np
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
from sensor import utils
from sensor.config import TARGET_COLUMN


class DataValidation:

    def __init__(self, data_validation_config:config_entity.DataValidationConfig, 
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            # creating a dictionary to update all validation error messages
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)
        

    def drop_cols_with_missing_values(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop columns which contains missing values above the specified threshold

        df: Accepts a pandas Dataframe
        threshold: Percentage criterion to drop a column
        ===========================================================================================================
        returns a pandas Dataframe if atleat a single column is left after dropping cols with missing values else None
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            # selecting column name which contains null values
            logging.info(f"Selecting column names which contains null values above the {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            # we can insert column names to be dropped in the validation_error dictionary by providing a suitable key
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(columns = list(drop_column_names), inplace=True)

            # return None if no columns left
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise SensorException(e, sys)


    def whether_required_cols_exists(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        # check whether all cols are available in dataset or not
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_col in base_columns:
                if base_col not in current_columns:
                    logging.info(f"Column: [{base_col} is not available]")
                    missing_columns.append(base_col)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                # since there are columns missing, so we return False as output of this function which is designed to return a boolean
                return False
            # if all columns present then only we return True
            return True
        except Exception as e:
            raise SensorException(e, sys)


    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        # to prepare data drift report, doesn't returns anything
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_col in base_columns:
                base_data, current_data = base_df[base_col], current_df[base_col]
                # Null Hypothesis is that both the columns are drawn from the same distribution
                logging.info(f"Hypothesis {base_col}: {base_data.dtype},{current_data.dtype}")
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # We will accept the Null Hypothesis
                    drift_report[base_col] = {"pvalues":float(same_distribution.pvalue),"same_distribution":True}
                else:
                    # We have to reject the Null Hypothesis
                    drift_report[base_col] = {"pvalues":float(same_distribution.pvalue),"same_distribution":False}

            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise SensorException(e, sys)



    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN}, inplace=True)
            logging.info("Replace na values with NAN in the base dataframe")
            base_df = self.drop_cols_with_missing_values(df=base_df, report_key_name="missing_values_within_base_dataset")

            logging.info("Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Drop columns having null values from train dataframe")
            train_df = self.drop_cols_with_missing_values(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info("Drop columns having null values from test dataframe")
            test_df = self.drop_cols_with_missing_values(df=test_df, report_key_name="missing_values_within_test_dataset")

            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_columns_to_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_to_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_to_float(df=test_df, exclude_columns=exclude_columns)

            logging.info("Checking whether all required columns present in train dataframe")
            train_df_columns_status = self.whether_required_cols_exists(base_df=base_df, current_df=train_df, 
            report_key_name = "missing_columns_within_train_dataset")
            logging.info("Checking whether all required columns present in test dataframe")
            test_df_columns_status = self.whether_required_cols_exists(base_df=base_df, current_df=test_df, 
            report_key_name = "missing_columns_within_test_dataset")

            if train_df_columns_status:
                # if above returns True then execute the following
                logging.info("Since all columns are present in train dataframe, hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info("Since all columns are present in test dataframe, hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")

            # writing the report in a yaml file created in utils.py
            logging.info("Writing report in validation yaml file")
            utils.write_yaml_file(file_path = self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

  
        except Exception as e:
            raise SensorException(e, sys)