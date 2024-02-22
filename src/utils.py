import os
import sys
import dill
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category = ConvergenceWarning)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error occured at save object")
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5, verbose=True)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_model_accuracy_score=accuracy_score(y_test, y_pred.round())
            report[list(models.keys())[i]]=test_model_accuracy_score

        return report

    except Exception as e:
        logging.info("Error occured at evaluate model")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)