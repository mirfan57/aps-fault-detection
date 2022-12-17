# task is to read the latest model (currently being used for production) and make prediction

import os, sys
from typing import Optional
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.config_entity import MODEL_FILE_NAME, TRANSFORMER_OBJ_FILE_NAME, TARGET_ENCODER_OBJ_FILE_NAME
from glob import glob     # returns all the files that we have inside folder

class ModelResolver:
# will give latest locations for model object, transformer, target_encoder; load them and compare to our recently trained model for comparison

    def __init__(self, model_registry:str = "saved_models", transformer_dir_name = "transformer",
                target_encoder_dir_name = "target_encoder", model_dir_name = "model"):
        # will ask for saved model folder (model_registry) location
        logging.info(f"Entered Model Resolver class")
        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)      # creates directory named saved_models
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name
    
    # get path of latest folder location
    def get_latest_dir_path(self) -> Optional[str]:
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            # the directory names are all in strings, we have to convert them to integer like '0' to 0, '1' to 1
            dir_names = list(map(int, dir_names))
            # to get the latest directory name
            latest_dir_name = max(dir_names)
            logging.info(f"Fetching latest directory path")
            return os.path.join(self.model_registry, f"{latest_dir_name}")
        except Exception as e:
            raise SensorException(e, sys)

    # in realtime we will have multiple models, transformers, label_encoders

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
            # say latest_dir = saved_models/1, self.model_dir_name=models, MODEL_FILE_NAME=model.pkl
            # returns saved_models/1/models/model.pkl
            logging.info(f"Fetching latest model path")
        except Exception as e:
            raise e    

    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJ_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Target encoder is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJ_FILE_NAME)
        except Exception as e:
            raise e

    # we want to save our model after comparison into a new dir incremented by 1 from latest dir number
    # we will write functions which give path to the location in saved_models dir where we will move our generated models from artifact dir
    # as they will be utilised in the prediction pipeline

    def get_latest_save_dir_path(self)->str:
        try:
        
            latest_dir = self.get_latest_dir_path()
            # for first iteration, when there is no dir inside saved_models then we will pass first file number as 0
            if latest_dir == None:
                logging.info(f"As there is nothing present inside {self.model_registry}, so creating a folder {0}")
                return os.path.join(self.model_registry, f"{0}")
            # for saved_models/1 path, os.path.basename returns 1
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            # for file already present inside saved_models then increment its basename by 1
            return os.path.join(self.model_registry, f"{latest_dir_num+1}")
            logging.info(f"Fetching path to save files after comparison (incrementing already present directory number)")
        except Exception as e:
            raise e

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
            logging.info(f"Fetching location to store best model after comparison")
        except Exception as e:
            raise e

    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJ_FILE_NAME)
            logging.info(f"Fetching location to store best transformer after comparison")
        except Exception as e:
            raise e

    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJ_FILE_NAME)
            logging.info(f"Fetching location to store best target encoder after comparison")
        except Exception as e:
            raise e
