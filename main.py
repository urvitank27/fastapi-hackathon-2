from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn, search_query_evaluator, gaussianNB
#import test_scraping

app = FastAPI()

@app.get("/ping")
async def ping() :
    return "Hello, I am alive"

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "Error": "Something went wrong"}),
)

@app.post("/search")
@app.get("/search")
async def search(query : str) :
    return search_query_evaluator.search_results(query)

@app.post("/predict")
@app.get("/predict")
async def predict(query : str) :
    return gaussianNB.predictDisease(query)

"""
@app.post("/find")
@app.get("/find")
async def find(lat : str, lon : str, search_type : str) :
    return test_scraping.find_nearest(lat, lon, search_type)
"""

if __name__ == "__main__" :
    uvicorn.run(app, host="localhost", port=8000)
