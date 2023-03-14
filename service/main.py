# from typing import Union
from fastapi.responses import RedirectResponse
from fastapi import FastAPI

from service.api.api import main_router

app = FastAPI()
app.include_router(main_router)

@app.get("/")
def read_root():
    response = RedirectResponse(url='/docs')
    return response


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}