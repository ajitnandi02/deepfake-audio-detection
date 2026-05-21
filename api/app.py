from fastapi import FastAPI, UploadFile, File
import shutil

app = FastAPI()

@app.get("/")
def home():

    return {"message":"Deepfake Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    path = "temp.wav"

    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    result = "Fake"

    return {"prediction":result}