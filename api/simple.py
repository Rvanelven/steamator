from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import requests
from io import BytesIO
from io import StringIO
import math

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pour lancer l'api
# utiliser make run_api
@app.get('/predict_tags')
def predict_tags(tags: str):
    io = StringIO(tags)
    tags_list = json.load(io)

    url_nlp = "https://storage.googleapis.com/wagon-data-770-vanelven/models/steamator/nlpmodel.joblib"
    url_vectorized_nlp = "https://storage.googleapis.com/wagon-data-770-vanelven/models/steamator/nlpvectorizermodel.joblib"

    # NLP
    response_nlp = requests.get(url_nlp).content
    mfile_nlp = BytesIO(response_nlp)
    model_nlp = joblib.load(mfile_nlp)

    print("nlp model loaded...")
    # NLP VECTORIZED
    response_vectorized_nlp = requests.get(url_vectorized_nlp).content
    mfile_vectorized_nlp = BytesIO(response_vectorized_nlp)
    model_vectorized_nlp = joblib.load(mfile_vectorized_nlp)
    print("nlp vectorized model loaded...")

    print("using both models...")
    print('vectorising tag list...')
    tags_vectorized = model_vectorized_nlp.transform([' '.join(tags_list)])
    print("transforming nlp")
    topic_proba_tags = model_nlp.transform(tags_vectorized)
    print("flatening result...")
    topic_proba_tags = [item for sublist in topic_proba_tags for item in sublist]
    print('transforming array to dict....')
    topic_proba_dict = dict(
            zip([
                'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5',
                'topic_6', 'topic_7', 'topic_8', 'topic_9', 'topic_10', 'topic_11', 'topic_12',
                'topic_13', 'topic_14', 'topic_15', 'topic_16', 'topic_17',
                'topic_18', 'topic_19'
            ], zip(topic_proba_tags)))

    topic_proba_dict = dict(
        map(lambda kv: (kv[0], kv[1][0]), topic_proba_dict.items()))

    return topic_proba_dict

# TODO: Add english (0-1) / has_a_website (0-1) / followers (integer)
@app.get("/")
def predict(price: int, topic_proba: str, short_desc: str, english: bool, has_a_website: bool, followers: int, nb_game_by_developer: int, days_on_steam=365):
    #passage de topic proba à un DF
    io = StringIO(topic_proba)
    topic_proba_dict = json.load(io)
    topic_proba_df = pd.DataFrame(topic_proba_dict, index=['1'])

    #short_desc passage à score
    top_100_words = ' discover first studio online adventure experience play high fly build use unique rpg escape bus explore tactical love chinese time classic bestcustomizeamp way game gight card mansion journey robot mysterious take style combat beat vr element friend creature underground secret unforgiving find fps mode hope battle part level novel survival others shooter world universe horror school music others girl story try evil system steam theme visual one going living dragon allows ultimate begin human new event player powerful original action life danmalu character turnbased weapon become three everything dungeon park touhou capture simulator create different multiplayer'
    top_100_words = top_100_words.split()
    score_descriptif = 0
    short_desc = short_desc.split()
    for word in short_desc:
        if word in top_100_words:
            score_descriptif += 1

    #english et has_a_website from bool to int
    english=int(english)
    has_a_website=int(has_a_website)
    #creation de la ligne de prediction sous forme de DF
    dict_to_transform = {"price":price, "score_descriptif": score_descriptif, "english":english, "has_a_website": has_a_website, "followers":followers, "nb_game_by_dev":nb_game_by_developer, "days_on_steam":days_on_steam}
    avant_dernier_df = pd.DataFrame(dict_to_transform, index=['1'])
    df_to_predict = avant_dernier_df.join(topic_proba_df)



    # TODO: import model class and make it work !
    url_final_model = "https://storage.googleapis.com/wagon-data-770-vanelven/models/steamator/model.joblib"
    response_final_model = requests.get(url_final_model).content
    mfile_final_model = BytesIO(response_final_model)
    final_model = joblib.load(mfile_final_model)
    print("final model loaded...")
    print("launching predict...")
    log_nbr_review = final_model.predict(df_to_predict)
    print('predict ok...')


    owner_estimated = int(math.exp(log_nbr_review[0]) * 130.77)
    print({"owner_estimated": owner_estimated * 365 / 647, "slope": score_descriptif / 100, "intercept":followers / 3, "price":price })


    return ({
        "owner_estimated": owner_estimated * 365 / 647,
        "slope": score_descriptif / 100,
        "intercept": followers / 3,
        "price": price
    })
