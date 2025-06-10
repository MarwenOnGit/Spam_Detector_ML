from fastapi import FastAPI
from pydantic import BaseModel
from Spam_Detector import predict

app = FastAPI()

class Email(BaseModel):
    message: str

@app.post("/test")
def predict(email: Email):
    return {"prediction": predict(email.message)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
