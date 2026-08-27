from fastapi import FastAPI

app = FastAPI() #creates your backend application.


@app.get("/health") #This is a simple API endpoint 
def health():
    return {"status": "ok"} #checks if the server is running