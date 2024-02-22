import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format="[ %(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s]"
                    )


project_name = "AppleQualityAnalysis"

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/pipeline_prediction.py",
    "src/pipeline/pipeline_training.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "notebook/data",
    "notebook/EDA.ipynb",
    "notebook/model.ipynb",
    "templates/index.html",
    "templates/home.html",
    "app.py",
    "requirements.txt",
    "setup.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    
    if(not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")

    
    else:
        logging.info(f"{filename} is already created")
