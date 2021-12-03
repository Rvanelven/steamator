from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def predict():
    return {"owner": 10293}
