from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def predict(price: int, tags: str, short_desc: str):

    tag_array = json.loads(tags)


    # TODO: preprocess the description to get the score
    # TODO: make a predict with tags, desc_score, price


    owner=12000
    slope=30
    intercept=0
    CA = owner * price
    return {"owner": owner, "slope": slope, "intercept":intercept, "Chiffre d'affaire": CA }
