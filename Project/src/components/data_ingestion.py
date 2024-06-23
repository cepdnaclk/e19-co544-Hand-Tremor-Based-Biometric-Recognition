import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

def remove_outliers_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def calculate_metrics(window):
    mean = window.mean()
    std_dev = window.std()
    energy = np.sum(np.square(window))
    hist = np.histogram(window, bins=10, density=True)[0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    peaks, _ = find_peaks(window, height=0)
    num_peaks = len(peaks)
    return mean, std_dev, energy, entropy, num_peaks

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('artifacts\dataset.csv')
            logging.info('Read the dataset as dataframe and initiate applying window extraction')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            person1 = df[df['ClassLabel'] == 1]
            person2 = df[df['ClassLabel'] == 2]
            person3 = df[df['ClassLabel'] == 2]
            person4 = df[df['ClassLabel'] == 4]
            person5 = df[df['ClassLabel'] == 5]

            window_size = 100
            all_metrics = []
            personas = [person1, person2, person3, person4, person5]

            for index, person_df in enumerate(personas):
                # Remove outliers for each column
                for col in ['X', 'Y', 'Z', 'Mixed']:
                    person_df = remove_outliers_iqr(person_df, col)

                # Calculate metrics for each column
                metrics = {}
                for col in ['X', 'Y', 'Z', 'Mixed']:
                    for metric_name in ['Mean', 'Std Dev', 'Energy', 'Entropy', 'Peaks']:
                        metrics[f'{metric_name}_{col}'] = []

                for i in range(0, len(person_df)):
                    for col in ['X', 'Y', 'Z', 'Mixed']:
                        window = person_df[col].iloc[i:i + window_size]
                        mean, std_dev, energy, entropy, num_peaks = calculate_metrics(window)
                        metrics[f'Mean_{col}'].append(mean)
                        metrics[f'Std Dev_{col}'].append(std_dev)
                        metrics[f'Energy_{col}'].append(energy)
                        metrics[f'Entropy_{col}'].append(entropy)
                        metrics[f'Peaks_{col}'].append(num_peaks)

                result_df = pd.DataFrame(metrics)
                result_df['category'] = index + 1  # Add category based on index (+1 to start from 1)
                all_metrics.append(result_df)

            # Concatenate all DataFrames into one
            combined_df = pd.concat(all_metrics)

            # Save combined DataFrame to CSV
            combined_df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Statistical extraction finished")

            #df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))