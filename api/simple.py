from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

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


# TODO: Add english (0-1) / has_a_website (0-1) / followers (integer)
@app.get("/")
def predict(price: int, tags: str, short_desc: str, english: bool, has_a_website: bool, followers: int, nb_game_by_developer: int):

    # TODO: import model class and make it work !
    # TODO: preprocess the description to get the score
    # TODO: make a predict with tags, desc_score, price

    #
    owner=12000
    slope= 0.5 # alpha / 0-1 sur scorer tags + desc
    intercept=0 # beta / 0-1 sur followers
    # price


    return {"owner": owner, "slope": slope, "intercept":intercept }
