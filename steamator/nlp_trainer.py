import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from google.cloud import storage
from sklearn.pipeline import make_pipeline


BUCKET_NAME = 'wagon-data-770-vanelven'
BUCKET_TRAIN_DATA_PATH = 'data/data_final_final_final.csv'
MODEL_NAME = 'steamator'
MODEL_VERSION = 'v1'
STORAGE_LOCATION_NLP = 'models/steamator/nlpmodel.joblib'
STORAGE_LOCATION_NLP_Vectorizer = 'models/steamator/nlpvectorizermodel.joblib'





class TrainerNlp():
    def __init__(self):
        self.pipeline = None

    def set_pipeline_nlp(self):
        print("setting the pipeline...")
        model = LatentDirichletAllocation(n_components=20)
        self.pipeline = make_pipeline(model)
        print("pipeline done")

    def get_list_of_topics(self):
        df = get_data_gcp()
        print('dataframe created from csv..')
        list_of_tags = df['top_5_tags'].tolist()
        self.vectorizer = TfidfVectorizer().fit(list_of_tags)
        self.data_vectorized = self.vectorizer.transform(list_of_tags)

        print(self.data_vectorized)

    def run(self):
        """set the pipeline"""
        self.set_pipeline_nlp()
        """ get the vectorized data """
        self.get_list_of_topics()
        print("training the model")
        self.pipeline.fit(self.data_vectorized)
        print("training done")

    def nlp_model_tags(self, df):

        nlp_vectors = self.pipeline.transform(df['top_5_tags'])
        df_with_20_probas = pd.DataFrame(data=nlp_vectors)
        df_with_20_probas = df_with_20_probas.rename(
            columns={
                0: "topic_0",
                1: "topic_1",
                2: "topic_2",
                3: "topic_3",
                4: "topic_4",
                5: "topic_5",
                6: "topic_6",
                7: "topic_7",
                8: "topic_8",
                9: "topic_9",
                10: "topic_10",
                11: "topic_11",
                12: "topic_12",
                13: "topic_13",
                14: "topic_14",
                15: "topic_15",
                16: "topic_16",
                17: "topic_17",
                18: "topic_18",
                19: "topic_19"
            })
        df = df.join(df_with_20_probas)
        return df

    # #def evaluate(self, X_test, y_test):
    #     """evaluates the pipeline on df_test and return the RMSE"""
    #     y_pred = self.pipeline.predict(X_test)
    #     rmse = compute_rmse(y_pred, y_test)
    #     return round(rmse, 2)

    def save_model_nlp(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'nlpmodel.joblib')
        print("saved nlpmodel.joblib locally")

        # Implement here
        self.__upload_nlpmodel_to_gcp()
        print(
            f"uploaded nlpmodel.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_NLP}"
        )

    def save_model_nlp_vectorized(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.vectorizer, 'nlpvectorizedmodel.joblib')
        print("saved nlpvectorizedmodel.joblib locally")

        # Implement here
        self.__upload_nlpmodel_vectorized_to_gcp()
        print(
            f"uploaded nlpmodel.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_NLP_Vectorizer}"
        )

    def __upload_nlpmodel_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION_NLP)
        blob.upload_from_filename('nlpmodel.joblib')

    def __upload_nlpmodel_vectorized_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION_NLP_Vectorizer)
        blob.upload_from_filename('nlpvectorizedmodel.joblib')


def get_data_gcp():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    return df


if __name__ == '__main__':
    # get training data from GCP bucket
    #df = get_data_gcp()

    # preprocess data
    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    trainer_nlp = TrainerNlp()

    trainer_nlp.run()
    # rmse = trainer.evaluate(X_test, y_test)
    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    trainer_nlp.save_model_nlp()
    trainer_nlp.save_model_nlp_vectorized()
