from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    # we want to know the location of imported data (feature_store_file_path), training(train_file_path) and testing(test_file_path) file
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str
    # we are providing convenience for reading our data, training or testing files

@dataclass
class DataValidationArtifact:
    report_file_path:str

@dataclass    
class DataTransformationArtifact:
    transform_object_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_encoder_path:str


class ModelTrainerArtifact:...
class ModelEvaluationArtifact:...
class ModelPusherArtifact:...