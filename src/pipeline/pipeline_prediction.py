import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path='artifact\model.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                Size: float,
                Weight: float,
                Sweetness: float,
                Crunchiness: float,
                Juiciness: float,
                Ripeness: float,
                Acidity: float
                ):
        
        self.Size =  Size
        self.Weight = Weight
        self.Sweetness = Sweetness
        self.Crunchiness = Crunchiness
        self.Juiciness = Juiciness
        self.Ripeness = Ripeness
        self.Acidity = Acidity
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                    'Size':[self.Size],
                    'Weight':[self.Weight],
                    'Sweetness':[self.Sweetness],
                    'Crunchiness':[self.Crunchiness],
                    'Juiciness':[self.Juiciness],
                    'Ripeness' :[self.Ripeness],
                    'Acidity' :[self.Acidity],
                    }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)