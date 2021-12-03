from steamator.data import get_data, clean_data
from steamator.utils import compute_rmse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
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
BUCKET_TRAIN_DATA_PATH = 'data/data_final.csv'

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
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

        #TO DO
    def set_pipeline(self):
        print("setting the pipeline...")
        forest = RandomForestRegressor(n_estimators=100, max_leaf_nodes=1000, max_depth=50)
        KNN = KNeighborsRegressor(n_neighbors=10)
        lasso = Lasso(max_iter=5000,positive=True, fit_intercept=False, )
        GBR = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3)
        # #ensemble = StackingRegressor(estimators=[('GBR', GBR),
        #                                          ("lasso", lasso),
        #                                          ('forest', forest)],
        #                              final_estimator=lasso,
        #                              n_jobs=-1)
        self.pipeline = make_pipeline(forest)
        print("pipeline done")

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        print("training the model")
        self.pipeline.fit(self.X, self.y)
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


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    return df

def preprocess(df):
    """method that pre-process the data"""
    X_train = df.drop(columns=[
        "steam_appid", "name", "top_5_tags", "nb_review", "owner_estimated",
        "rating", "popularity",
        "average_playtime",
        "median_playtime"
    ])
    y_train = df['owner_estimated']
    return X_train, y_train


STORAGE_LOCATION = 'models/steamator/model.joblib'

if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()

    # preprocess data
    X_train, y_train = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    trainer = Trainer(X_train, y_train)

    trainer.run()
    # rmse = trainer.evaluate(X_test, y_test)
    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    trainer.save_model()

#if __name__ == "__main__":
# df=get_data()
# df = clean_data(df)
# y=df["owner_median"]
# X = df.drop('owner_median', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# trainer = Trainer(X_train, y_train)
# trainer.run()
# rmse = trainer.evaluate(X_test, y_test)
# print(f"rmse: {rmse}")
