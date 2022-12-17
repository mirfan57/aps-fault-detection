import os, sys
from sensor.entity import config_entity, artifact_entity
from sensor.predictor import ModelResolver
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact
from sensor.logger import logging
from sensor.utils import load_object, save_object
from sensor.entity.config_entity import ModelPusherConfig


class ModelPusher:

    def __init__(self, model_pusher_config:config_entity.ModelPusherConfig, 
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_models_dir)
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_pusher(self,) -> ModelPusherArtifact:
        try:
            # load object
            logging.info(f"Loading current trained model, its transformer and target encoder objects")
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # model pusher dir (saving objects inside artifact directory)
            logging.info(f"Saving model inside artifact directory")
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            # above trained model object will be saved at location artifact/model_pusher/saved_models/model.pkl
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            # similarly transformer obj saved at location artifact/model_pusher/saved_models/transformer.pkl
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)
            # similarly target_encoder obj saved at location artifact/model_pusher/saved_models/target_encoder.pkl

            # saving objects at root location in saved_models dir
            # NOTE that we are first getting locations for each object and then we move to the saving step
            logging.info(f"Saving model in saved_models directory")
            # model obj to be saved in saved_models/{latest_num}+1/model/model.pkl
            model_path = self.model_resolver.get_latest_save_model_path()
            # transformer obj to be saved in saved_models/{latest_num}+1/transformer/transformer.pkl
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            # target_encoder obj to be saved in saved_models/{latest_num}+1/target_encoder/target_encoder.pkl
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path=model_path, obj=model)
            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=target_encoder_path, obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir, 
            saved_model_dir=self.model_pusher_config.saved_models_dir)
            # returns the path of saved_models folder location at root dir and artifact/model_pusher directory
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e, sys)