from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
def POST_predict():
    return {"message": "Hello"}
