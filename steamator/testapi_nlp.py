import joblib
import requests
from io import BytesIO
from io import StringIO
import json


url_nlp = "https://storage.googleapis.com/wagon-data-770-vanelven/models/steamator/nlpmodel.joblib"
url_vectorized_nlp = "https://storage.googleapis.com/wagon-data-770-vanelven/models/steamator/nlpvectorizermodel.joblib"
tags_string = "['action', 'drama', 'golf', 'movie', 'ninja']"

io = StringIO('["action", "drama", "golf", "movie", "ninja"]')

tags_list = json.load(io)

# NLP
response_nlp = requests.get(url_nlp).content
mfile_nlp = BytesIO(response_nlp)
model_nlp = joblib.load(mfile_nlp)

# NLP VECTORIZED
response_vectorized_nlp = requests.get(url_vectorized_nlp).content
mfile_vectorized_nlp = BytesIO(response_vectorized_nlp)
model_vectorized_nlp = joblib.load(mfile_vectorized_nlp)

tags_vectorized = model_vectorized_nlp.transform([' '.join(tags_list)])
print(tags_vectorized)
topic_proba_tags = model_nlp.transform(tags_vectorized)
print(topic_proba_tags)
topic_proba_tags = [item for sublist in topic_proba_tags for item in sublist]

topic_proba_dict = dict(
    zip([
        'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5',
        'topic_6', 'topic_7', 'topic_8', 'topic_9', 'topic_10', 'topic_11',
        'topic_12', 'topic_13', 'topic_14', 'topic_15', 'topic_16', 'topic_17',
        'topic_18', 'topic_19'
    ], zip(topic_proba_tags)))

topic_proba_dict = dict(
    map(lambda kv: (kv[0], kv[1][0]), topic_proba_dict.items()))
print(topic_proba_dict)
