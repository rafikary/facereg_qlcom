# main.py – API HRIS Face Recognition

from typing import List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== STARTUP ==================
@app.on_event("startup")
async def startup_event():
    engine.load_liveness_model()
# ============================================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "QLCOM Face Recognition API is active",
        "endpoints": {
            "register": "/register",
            "recognize": "/recognize",
            "recognize_multi": "/recognize_multi",
            "list_users": "/users",
            "delete_user": "/user/{employee_id}",
            "docs_swagger": "/docs"
        }
    }

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

@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    """
    - Normal: return engine.recognize_face() result
    - Jika wajah tidak terdeteksi / crop gagal: return 200 dengan manual_required=true
    """
    img_bytes = await image.read()

    try:
        result = engine.recognize_face(img_bytes)

        # default: manual tidak diperlukan (jangan override kalau engine sudah set)
        if isinstance(result, dict):
            result.setdefault("manual_required", False)
            result.setdefault("manual_reason", None)
            result.setdefault("manual_allowed", False)

        return {"status": "success", "data": result}

    except ValueError as e:
        # Kasus umum: tidak ada wajah / gagal decode / enforce_detection
        msg = str(e)
        msg_l = msg.lower()

        # Kalau file benar-benar rusak / bukan image => ini error input (400), bukan manual
        if "gagal membaca gambar" in msg_l:
            raise HTTPException(status_code=400, detail=msg)

        face_not_detected = (
            "gagal mendeteksi wajah" in msg_l
            or "tidak ada wajah ditemukan" in msg_l
            or "enforce_detection" in msg_l
        )

        return {
            "status": "success",
            "data": {
                "match": False,
                "reason": "face_not_detected" if face_not_detected else "error",
                "detail": msg,
                "manual_required": face_not_detected,
                "manual_reason": "face_not_detected" if face_not_detected else "error",
                "manual_allowed": face_not_detected
            }
        }

    except Exception as e:
        # error selain face-not-detected -> 500
        raise HTTPException(status_code=500, detail=str(e))

# OPTIONAL: multi-frame voting
@app.post("/recognize_multi")
async def recognize_multi(images: List[UploadFile] = File(...)):
    try:
        imgs = [await f.read() for f in images]
        result = engine.recognize_face_multi(imgs)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users")
async def list_users():
    if not os.path.exists(DB_PATH):
        return {"status": "success", "data": []}

    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    users = [
        {
            "employee_id": emp_id,
            "name": data.get("name"),
            "num_embeddings": len(data.get("embeddings", []) or []),
        }
        for emp_id, data in db.items()
    ]

    return {"status": "success", "data": users}

@app.delete("/user/{employee_id}")
async def delete_user(employee_id: str):
    result = engine.delete_user(employee_id)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail=result.get("reason", "User tidak ditemukan"))
    return {"status": "success", "data": result}
