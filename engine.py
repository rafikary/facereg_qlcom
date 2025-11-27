# engine.py - ArcFace + MTCNN (DeepFace 0.0.96 compatible)

import json
import os
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from deepface import DeepFace

DB_PATH = "faces_db.json"
DETECTOR_BACKEND = "mtcnn"  

# ================== UTIL DB ==================

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r") as f:
        return json.load(f)


def save_db(db: Dict[str, Any]) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


# ================== UTIL IMAGE ==================

def bytes_to_bgr_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan file jpg/png/jpeg valid.")
    return img


# ================== ARCFace Embedding ==================

def get_arcface_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Mengambil embedding wajah (512 dim) menggunakan ArcFace + MTCNN.
    """
    img = bytes_to_bgr_image(image_bytes)

    try:
        reps = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
    except Exception as e:
        raise ValueError(f"Gagal mendeteksi wajah: {e}")

    if not reps:
        raise ValueError("Tidak ada wajah ditemukan.")

    emb = reps[0]["embedding"]
    return np.array(emb, dtype=np.float32)


# ================== REGISTER ==================

def register_face(employee_id: str, name: str, image_bytes: bytes) -> Dict[str, Any]:
    db = load_db()

    emb = get_arcface_embedding(image_bytes)
    emb_list = emb.tolist()

    if employee_id in db:
        db[employee_id]["embeddings"].append(emb_list)
        db[employee_id]["name"] = name
    else:
        db[employee_id] = {
            "name": name,
            "embeddings": [emb_list]
        }

    save_db(db)

    return {
        "employee_id": employee_id,
        "name": name,
        "num_embeddings": len(db[employee_id]["embeddings"])
    }

# ================== RECOGNIZE ==================

def recognize_face(image_bytes: bytes, threshold: float = 3.5) -> Dict[str, Any]:
    """
    Mencocokkan wajah dengan database.
    threshold default 3.5 (disesuaikan untuk masker + kamera laptop).
    """
    db = load_db()
    if not db:
        return {"match": False, "reason": "Database kosong."}

    query_emb = get_arcface_embedding(image_bytes)

    best_id: Optional[str] = None
    best_name: Optional[str] = None
    best_dist: float = 999.0

    # cari jarak terdekat
    for emp_id, data in db.items():
        name = data["name"]
        for emb in data["embeddings"]:
            emb_vec = np.array(emb, dtype=np.float32)
            dist = np.linalg.norm(query_emb - emb_vec)

            if dist < best_dist:
                best_dist = dist
                best_id = emp_id
                best_name = name

    # ---------- MATCH ----------
    if best_id is not None and best_dist < threshold:

        # Confidence level (berbasis eksperimen: masker, low light)
        if best_dist < 2.0:
            confidence = "high"      # sangat mirip
        elif best_dist < 3.5:
            confidence = "medium"    # mirip, pengaruh masker/cahaya
        else:
            confidence = "low"

        return {
            "match": True,
            "employee_id": best_id,
            "name": best_name,
            "distance": float(best_dist),
            "threshold": threshold,
            "confidence": confidence
        }

    # ---------- TIDAK MATCH ----------
    return {
        "match": False,
        "reason": "Tidak ada wajah cocok.",
        "best_distance": float(best_dist),
        "threshold": threshold
    }


# ================== DELETE USER ==================

def delete_user(employee_id: str) -> Dict[str, Any]:
    db = load_db()

    if employee_id not in db:
        return {"deleted": False, "reason": "User tidak ditemukan"}

    del db[employee_id]
    save_db(db)

    return {"deleted": True, "employee_id": employee_id}
