import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


def predict(features):
    try:
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
        logging.info("Loading models...")
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        logging.info("Loaded models")
        data_scaled=preprocessor.transform(features)
        preds=model.predict(data_scaled)
        logging.info("Predictions made")
        return preds
    
    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)



