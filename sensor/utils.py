import os,sys
import numpy as np
import pandas as pd
import yaml
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import mongo_client


def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        # find() gives records of all documents inside collection but is a generator which is passed to the list and converted to dataframe
        logging.info(f"Found columns: {df.columns}")
        # we know that mongodb has an unecessary _id column which is not important and need to be dropped
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id")
            df.drop("_id", axis=1, inplace=True)
        logging.info(f"Rows and cols in df: {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e, sys)


def write_yaml_file(file_path, data:dict):
    # to save report in yaml format
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok = True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise SensorException(e, sys)


def convert_columns_to_float(df:pd.DataFrame, exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise SensorException(e, sys)