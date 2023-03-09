from fastapi import APIRouter

from service.api.endpoints.detect import yolo_router
from service.api.endpoints.test import test_router

main_router = APIRouter()

main_router.include_router(yolo_router)
main_router.include_router(test_router)


