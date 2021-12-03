from fastapi import FastAPI
app = FastAPI()

# pour lancer l'api
# utiliser make run_api

@app.get("/")
def predict():
    return {"owner": 12000}
