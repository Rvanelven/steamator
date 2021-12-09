# from steamator.data import get_data, clean_data
from steamator.utils import compute_rmse
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from google.cloud import storage
import pandas as pd
import numpy as np
import joblib

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-770-vanelven'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/data_final_final_final.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'steamator'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

class Trainer():
    def __init__(self, X_train, X_test, y_train, y_test):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def set_pipeline(self):
        print("setting the pipeline...")
        num_transformer = make_pipeline(StandardScaler())
        preproc = make_column_transformer(
            (num_transformer, make_column_selector(dtype_include=['float64'])),
            remainder='passthrough')
        model = RandomForestRegressor(bootstrap=True,
                                       max_features=0.4,
                                       min_samples_leaf=14,
                                       n_estimators=100,
                                       min_samples_split=14)
        self.pipeline = make_pipeline(preproc, model)
        print("pipeline done")

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        print("training the model")
        self.pipeline.fit(self.X_train, self.y_train)
        print("training done")

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 2)

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.__upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

    def __upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')




def get_data_gcp():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    return df

def preprocess(df):
    """method that pre-process the data"""
    X= df.drop(columns=['steam_appid', 'top_5_tags',
                                'owner_estimated',
                                'nb_review', 'target'

                               ])
    y = df["target"]
    X = pd.DataFrame(X)
    X_train, y_train, X_test, y_test = train_test_split(X,y,test_size = 0.3)
    return X_train, y_train, X_test, y_test


STORAGE_LOCATION = 'models/steamator/model.joblib'

if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data_gcp()

    # preprocess data
    X_train, y_train, X_test, y_test = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    trainer = Trainer(X_train, y_train, X_test, y_test)

    trainer.run()
    # rmse = trainer.evaluate(X_test, y_test)
    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    trainer.save_model()
