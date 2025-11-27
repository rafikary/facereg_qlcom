# main.py – API HRIS Face Recognition

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import engine
import json
import os

DB_PATH = "faces_db.json"

app = FastAPI(
    title="QLCOM Face Recognition API",
    description="API untuk register & recognize wajah karyawan",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== ROOT (Homepage API) ==================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "QLCOM Face Recognition API is active",
        "endpoints": {
            "register": "/register",
            "recognize": "/recognize",
            "list_users": "/users",
            "delete_user": "/user/{employee_id}",
            "docs_swagger": "/docs"
        }
    }


# ================== REGISTER ==================

@app.post("/register")
async def register(
    employee_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        img_bytes = await image.read()
        result = engine.register_face(employee_id, name, img_bytes)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ================== RECOGNIZE ==================

@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        result = engine.recognize_face(img_bytes)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ================== LIST USERS ==================

@app.get("/users")
async def list_users():

    if not os.path.exists(DB_PATH):
        return {"status": "success", "data": []}

    with open(DB_PATH, "r") as f:
        db = json.load(f)

    users = [
        {
            "employee_id": emp_id,
            "name": data["name"],
            "num_embeddings": len(data["embeddings"]),
        }
        for emp_id, data in db.items()
    ]

    return {"status": "success", "data": users}


# ================== DELETE USER ==================

@app.delete("/user/{employee_id}")
async def delete_user(employee_id: str):
    result = engine.delete_user(employee_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail=result["reason"])
    return {"status": "success", "data": result}
