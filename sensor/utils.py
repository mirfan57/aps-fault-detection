import os,sys
import numpy as np
import pandas as pd
import yaml
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import mongo_client
import dill


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

def save_object(file_path:str, obj:object) -> None:
    try:
        logging.info("Entered the save_object method of the utils class")
        os.makedirs(file_path, exist_ok = True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            # dill also saves object in a pickle file
        logging.info("Exited the save_object method of the utils class")
    except Exception as e:
        raise SensorException(e, sys) from e

def load_object(file_path:str,) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The given {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SensorException(e, sys) from e

# since transformation converts our dataframe into an array
# we need 2 functions: one to load numpy array into a file and a another to load the file as numpy array

def save_numpy_array_data(file_path:str, array:np.array):
    """
    Save numpy array data to file
    file_path: str_location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SensorException(e, sys) from e

def load_numpy_array_data(file_path:str) -> np.array:
    """
    Loads numpy array data from file
    file_path: str location of the file to load
    return: np.array of loaded data
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(e, sys)