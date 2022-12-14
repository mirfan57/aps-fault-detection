import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sensor import utils
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging


class DataIngestion:

    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig ):
        # DataIngestionConfig class is the input required for the DataIngestion component, so passed inside the constructor
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_data_ingestion(self) ->artifact_entity.DataIngestionArtifact:
        # output will be data_ingestion_artifact
        try:
            logging.info(f"Exporting data from my collection as pandas dataframe.")
            # exporting collection data as pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(database_name = self.data_ingestion_config.database_name,
                                                                collection_name = self.data_ingestion_config.collection_name)

            # now we have to save this dataframe in feature store
            logging.info("Saving data in feature store")

            # replace na with nan
            df.replace(to_replace="na", value = np.NAN, inplace=True)

            # Save data to feature store
            logging.info("Create feature store folder if not present")

            # create feature store folder if not available
            # os.path.dirname() method is used to get the directory name from the specified path.
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            # os.makedirs() method will create all unavailable/missing directory in the specified path.
            os.makedirs(feature_store_dir, exist_ok=True)      

            # save df as csv to feature store folder
            df.to_csv(path_or_buf = self.data_ingestion_config.feature_store_file_path, header=True, index=False)

            # splitting dataset into train and test set
            logging.info("Splitting our dataset into train and test set")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state = 12)

            logging.info("Create dataset directory folder if not available")
            # creating dataset directory folder if not present
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok = True)

            # Now saving df to above created/already present train_file_path
            logging.info("Save df as csv file in feature store folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            #Prepare artifact or output
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(feature_store_file_path = 
            self.data_ingestion_config.feature_store_file_path, train_file_path = self.data_ingestion_config.train_file_path, 
            test_file_path = self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            # returning the output (artifact)
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
