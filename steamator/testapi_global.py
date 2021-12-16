import json
import pandas as pd
import requests
from io import BytesIO
from io import StringIO
import math
import joblib

topic_proba = '''{
    "topic_0": 0.025000000861780884,
    "topic_1": 0.02500000077182249,
    "topic_2": 0.025000000951429034,
    "topic_3": 0.025000000225670182,
    "topic_4": 0.025000000516567672,
    "topic_5": 0.02500000093535779,
    "topic_6": 0.02500000023141724,
    "topic_7": 0.025000001000781195,
    "topic_8": 0.02500000055570622,
    "topic_9": 0.025000000985411636,
    "topic_10": 0.025000000985411636,
    "topic_11": 0.025000000042035665,
    "topic_12": 0.025000000147243833,
    "topic_13": 0.5249999899502142,
    "topic_14": 0.025000000498525642,
    "topic_15": 0.025000000123481702,
    "topic_16": 0.025000000218193184,
    "topic_17": 0.02500000081524997,
    "topic_18": 0.025000000080510877,
    "topic_19": 0.02500000029075559
}'''

short_desc = "action nudity gore golf"
english = 1
has_a_website = 1
price = 10
followers = 102
nb_game_by_developer = 2
days_on_steam=365

#passage de topic proba à un DF
io = StringIO(topic_proba)
topic_proba_dict = json.load(io)
topic_proba_df = pd.DataFrame(topic_proba_dict, index=["1"])
print(topic_proba_df)

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
avant_dernier_df = pd.DataFrame(dict_to_transform, index=["1"])
df_to_predict = avant_dernier_df.join(topic_proba_df)
print(df_to_predict)

#RECUP du modèle sur gcp et utilisation
url_final_model = "https://storage.googleapis.com/wagon-data-770-vanelven/models/steamator/model.joblib"
print('url ok...')
response_final_model = requests.get(url_final_model).content
print('response ok...')
mfile_final_model= BytesIO(response_final_model)
print('request ok...')
final_model = joblib.load(mfile_final_model)
print('final model ok...')

log_nbr_review = final_model.predict(df_to_predict)
print('predict ok...')
print(f"log_nbr_review= {log_nbr_review}")
owner_estimated = int(math.exp(log_nbr_review[0]) * 130.77)
print(f"owner_estimated= {owner_estimated}")
print(f"score_descriptif= {score_descriptif}")
print(f"followers= {followers}")

# TODO
# owner_estimated => owner_estimated * 365 / 647
# slope 0-1 score_desc
# intercept 0-1 followers


print({"owner_estimated": owner_estimated * 365 / 647, "slope": score_descriptif / 100, "intercept":followers / 3, "price":price })
