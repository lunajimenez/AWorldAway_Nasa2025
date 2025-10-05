from fastapi import APIRouter

router = APIRouter()


@router.get("/stats")
def POST_predict():
    return {"message": "Hello"}
