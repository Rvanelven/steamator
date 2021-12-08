import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from google.cloud import storage
from steamator.data import get_data


def nlp_model_tags(df):
    df=get_data()
    list_of_tags = df['top_5_tags'].tolist()
    vectorizer = TfidfVectorizer().fit(list_of_tags)
    data_vectorized = vectorizer.transform(list_of_tags)
    nlp_model = LatentDirichletAllocation(n_components=20).fit(data_vectorized)
    joblib.dump(nlp_model, 'nlpmodel.joblib')
    client = storage.Client()
    bucket = client.bucket('wagon-data-770-vanelven')
    blob = bucket.blob('models/steamator/nlpmodel.joblib')
    blob.upload_from_filename('nlpmodel.joblib')
    nlp_vectors = nlp_model.transform(df['top_5_tags'])
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
