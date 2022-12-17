import os, sys
import pandas as pd
import numpy as np
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.predictor import ModelResolver
from sensor.utils import load_object
from datetime import datetime
PREDICTION_DIR = "prediction"



def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        # creating model_resolver object by passing saved_models because we have saved our model there
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN}, inplace=True)

        logging.info(f"Loading transformer to transform input dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        # getting feature names
        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        logging.info(f"Loading target encoder to convert our predicted column to categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        categorical_prediction = target_encoder.inverse_transform(prediction)

        # update df with our numerical predictions and corresponding categorical labels
        df["prediction"] = prediction
        df["categorical_prediction"] = categorical_prediction

        # getting location to save our predictions --> prediction/{input_file_name}{timestamp}.csv
        prediction_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)

        # saving pur df in csv format at above file location
        df.to_csv(prediction_file_path, index=False, header = True)
        return prediction_file_path
    except Exception as e:
        SensorException(error_message=e, error_detail=sys)