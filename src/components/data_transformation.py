from src.logger import logging
from src.exception import CustomException
import os 
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation started.....')
            # define  which columns should be ordinal-encoded and which should be scaled
            categorical_columns = []
            numerical_columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness',
       'Acidity']

            # Numerical Pipeline

            logging.info('Pipeline is started.....')

            num_pipeline = Pipeline(
                steps=[
                    ('SimpleImputer',SimpleImputer(strategy='median')),
                    ('StandardScaler',StandardScaler())
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps=[
                    ('SimpleImputer',SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder',OrdinalEncoder())
                ]
            )

            # combine num_pipeline and cat_pipeline

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('categorical_pipeline',cat_pipeline,categorical_columns)
            ])
            

            logging.info('Pipeline is completed.....')

            return preprocessor

        except Exception as e:
            logging.info('Error occured at get data transformation object.....')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading is completed.....')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtainig preprocessing object.....')
            preprocessing_obj = self.get_data_transformation_object()


            target_column_name = 'Quality'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on Training and Testing data')

            # Transforming using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pikle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Exception occured in the initiate data transformation.....')
            raise CustomException(e,sys)