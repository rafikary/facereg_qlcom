# engine.py - ArcFace + MTCNN (DeepFace 0.0.96 compatible)
# + Liveness (OpenVINO anti-spoof-mn3 + optional DeepFace anti_spoofing)
import json
import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
from deepface import DeepFace
import openvino as ov

DB_PATH = "faces_db.json"
DETECTOR_BACKEND = "mtcnn"

# ================== LIVENESS CONFIG ==================
LIVENESS_MODEL_PATH = "models/anti-spoof-mn3.onnx"

# OpenVINO score band
OV_PASS = 0.55           # target "aman" (relaxed untuk real faces)
OV_UNCERTAIN_LOW = 0.45  # bawah ini -> nanggung / minta ulang
OV_STRONG_SPOOF = 0.30   # bawah ini -> spoof kuat

# DeepFace antispoof score band
DF_PASS = 0.85          # relaxed sedikit untuk kondisi lighting buruk
DF_STRONG_LIVE = 0.95   # super yakin hidup
DF_STRONG_SPOOF = 0.50  # score di bawah ini + flag spoof => spoof kuat

# Quality gate (lebih toleran untuk webcam/kondisi lighting tidak ideal)
MIN_FACE_AREA = 90 * 90
MIN_FACE_CONF = 0.70     # relaxed dari 0.80
MIN_BLUR_VAR = 10.0      # relaxed dari 15.0
MIN_BRIGHT = 15.0        # relaxed dari 25.0 (toleran backlight)
MAX_BRIGHT = 250.0       # relaxed dari 240.0

# Optional: enforce liveness on enrollment
ENROLL_REQUIRE_LIVENESS = True

# ================== MATCH CONFIG ==================
DEFAULT_THRESHOLD = 3.5
MATCH_MARGIN = 0.20

# OpenVINO globals
_OV_CORE: Optional[ov.Core] = None
_OV_COMPILED = None
_OV_OUTPUT = None


def load_liveness_model() -> None:
    """Call once at startup. If model missing => bypass liveness."""
    global _OV_CORE, _OV_COMPILED, _OV_OUTPUT

    if not os.path.exists(LIVENESS_MODEL_PATH):
        print(f"[LIVENESS] Model not found: {LIVENESS_MODEL_PATH} (bypass)")
        _OV_CORE = None
        _OV_COMPILED = None
        _OV_OUTPUT = None
        return

    _OV_CORE = ov.Core()
    model = _OV_CORE.read_model(LIVENESS_MODEL_PATH)
    _OV_COMPILED = _OV_CORE.compile_model(model, "CPU")
    _OV_OUTPUT = _OV_COMPILED.output(0)
    print(f"[LIVENESS] Loaded: {LIVENESS_MODEL_PATH}")


# ================== UTIL DB ==================

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db: Dict[str, Any]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


# ================== UTIL IMAGE ==================

def bytes_to_bgr_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan jpg/png/jpeg valid.")
    return img


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _expand_bbox(x: int, y: int, w: int, h: int, pct: int, W: int, H: int) -> Tuple[int, int, int, int]:
    # expand percentage applies to width/height total (split to both sides)
    dx = int((w * pct / 100.0) / 2.0)
    dy = int((h * pct / 100.0) / 2.0)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(W, x + w + dx)
    y2 = min(H, y + h + dy)
    return x1, y1, x2, y2


def _preprocess_ov(face_bgr_u8: np.ndarray) -> np.ndarray:
    """Return NCHW float32 1x3x128x128 with correct BGR mean/scale."""
    face = cv2.resize(face_bgr_u8, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    # RGB mean/scale from model zoo, reorder -> BGR
    mean_bgr = np.array([107.8395, 119.5950, 151.2405], dtype=np.float32)  # B,G,R
    scale_bgr = np.array([55.0035, 56.4570, 63.0105], dtype=np.float32)    # B,G,R
    face = (face - mean_bgr) / scale_bgr

    inp = np.transpose(face, (2, 0, 1))[None, ...]  # (1,3,128,128)
    return inp


def _infer_ov(face_bgr_u8: np.ndarray) -> Tuple[float, float]:
    """Return (p_real, p_spoof). Handles softmax if needed."""
    inp = _preprocess_ov(face_bgr_u8)
    out = _OV_COMPILED([inp])[_OV_OUTPUT]  # (1,2)
    out = np.array(out).reshape(2)
    p0, p1 = float(out[0]), float(out[1])

    s = p0 + p1
    if not (0.9 <= s <= 1.1) or p0 < 0 or p1 < 0 or p0 > 2 or p1 > 2:
        ex = np.exp(out - np.max(out))
        sm = ex / (np.sum(ex) + 1e-9)
        p0, p1 = float(sm[0]), float(sm[1])

    return p0, p1


def _enhance_lighting(face_bgr_u8: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    untuk normalize lighting dan handle backlight/underexposed images.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(face_bgr_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced


def _quality_metrics(face_bgr_u8: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(face_bgr_u8, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    return {"blur_var": blur_var, "bright": bright}


# ================== LIVENESS ==================

def detect_spoof(image_bytes: bytes) -> Dict[str, Any]:
    """
    Returns:
      {
        is_live: bool,
        reason: ok|spoof_detected|liveness_uncertain|quality_low|bypass,
        prob_real: float|None, prob_spoof: float|None,  (OpenVINO mean)
        df_is_real: bool|None, df_score: float|None,
        model: str,
        debug: str
      }
    """
    # Lazy load (optional)
    if _OV_COMPILED is None and os.path.exists(LIVENESS_MODEL_PATH):
        try:
            load_liveness_model()
        except Exception:
            pass

    if _OV_COMPILED is None:
        return {
            "is_live": True,
            "reason": "bypass",
            "prob_real": 1.0,
            "prob_spoof": 0.0,
            "df_is_real": None,
            "df_score": None,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": "bypass (OpenVINO model not loaded)"
        }

    img_bgr = bytes_to_bgr_image(image_bytes)
    H, W = img_bgr.shape[:2]

    # 1) Extract face once (try WITH anti_spoofing; if fail, fallback without)
    meta = None
    face_bgr = None
    df_is_real = None
    df_score = None

    try:
        faces = DeepFace.extract_faces(
            img_path=img_bgr,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=False,
            expand_percentage=20,
            color_face="bgr",
            anti_spoofing=True,  # needs torch
        )
        meta = faces[0]
        face_bgr = meta["face"]
        df_is_real = meta.get("is_real", None)
        df_score = _safe_float(meta.get("antispoof_score", None), None)
    except Exception as e:
        # fallback: still do OpenVINO only
        try:
            faces = DeepFace.extract_faces(
                img_path=img_bgr,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=False,
                expand_percentage=20,
                color_face="bgr",
            )
            meta = faces[0]
            face_bgr = meta["face"]
        except Exception as e2:
            return {
                "is_live": False,
                "reason": "quality_low",
                "prob_real": None,
                "prob_spoof": None,
                "df_is_real": None,
                "df_score": None,
                "model": "anti-spoof-mn3(+deepface)",
                "debug": f"crop fail: df={e} | fallback={e2}"
            }

    if face_bgr is None:
        return {
            "is_live": False,
            "reason": "quality_low",
            "prob_real": None,
            "prob_spoof": None,
            "df_is_real": None,
            "df_score": None,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": "empty face"
        }

    # face_bgr could be float 0..1
    face_np = np.array(face_bgr)
    if float(face_np.max()) <= 1.0:
        face_u8 = (face_np * 255.0).clip(0, 255).astype(np.uint8)
    else:
        face_u8 = face_np.clip(0, 255).astype(np.uint8)

    # 2) Quality gate
    fa = (meta or {}).get("facial_area", {}) or {}
    x = int(fa.get("x", 0) or 0)
    y = int(fa.get("y", 0) or 0)
    w = int(fa.get("w", 0) or 0)
    h = int(fa.get("h", 0) or 0)
    conf = _safe_float((meta or {}).get("confidence", 1.0), 1.0)

    qm = _quality_metrics(face_u8)
    blur_var = qm["blur_var"]
    bright = qm["bright"]
    area = w * h

    low_conf = conf is not None and conf < MIN_FACE_CONF
    too_small = area < MIN_FACE_AREA
    too_blur = blur_var < MIN_BLUR_VAR
    too_dark = bright < MIN_BRIGHT
    too_bright = bright > MAX_BRIGHT

    if low_conf or too_small or too_blur or too_dark or too_bright:
        return {
            "is_live": False,
            "reason": "quality_low",
            "prob_real": None,
            "prob_spoof": None,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": f"quality_low conf={conf:.2f} area={w}x{h} blur={blur_var:.1f} bright={bright:.1f}"
        }

    # 3) OpenVINO multi-crop inference from bbox (stabilizer + include background)
    # Apply CLAHE preprocessing untuk handle bad lighting
    ov_scores = []
    ov_spoofs = []

    if w > 0 and h > 0:
        for pct in (0, 20, 40):
            x1, y1, x2, y2 = _expand_bbox(x, y, w, h, pct, W, H)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Apply CLAHE preprocessing untuk normalize lighting
            crop_enhanced = _enhance_lighting(crop.astype(np.uint8))
            p_real, p_spoof = _infer_ov(crop_enhanced)
            ov_scores.append(p_real)
            ov_spoofs.append(p_spoof)
    else:
        # Apply CLAHE ke face crop juga
        face_enhanced = _enhance_lighting(face_u8)
        p_real, p_spoof = _infer_ov(face_enhanced)
        ov_scores.append(p_real)
        ov_spoofs.append(p_spoof)

    if not ov_scores:
        return {
            "is_live": False,
            "reason": "quality_low",
            "prob_real": None,
            "prob_spoof": None,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": "ov inference failed (no crops)"
        }

    ov_real = float(np.mean(ov_scores))
    ov_spoof = float(np.mean(ov_spoofs))

    # 4) Decision logic (lebih ketat ke spoof HP/monitor)

    # --- Hard spoof: OpenVINO sangat yakin spoof
    if ov_real <= OV_STRONG_SPOOF:
        return {
            "is_live": False,
            "reason": "spoof_detected",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": f"FAIL(ov_strong_spoof) ov_real={ov_real:.3f} df_is_real={df_is_real} df_score={df_score}"
        }

    # --- Hard spoof: DeepFace flag bilang spoof
    if df_is_real is False:
        # boolean dari DF kuat -> langsung spoof
        return {
            "is_live": False,
            "reason": "spoof_detected",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": f"FAIL(df_flag_spoof) ov_real={ov_real:.3f} df_is_real={df_is_real} df_score={df_score}"
        }

    # --- Strong-live shortcut:
    # kalau OpenVINO sudah sangat yakin, tapi DF skornya jelek -> jangan diloloskan
    if ov_real >= 0.90:
        if df_score is None:
            # Tidak ada DF, percaya OpenVINO
            return {
                "is_live": True,
                "reason": "ok",
                "prob_real": ov_real,
                "prob_spoof": ov_spoof,
                "df_is_real": None,
                "df_score": None,
                "model": "anti-spoof-mn3",
                "debug": f"PASS(ov_strong_live_no_df) ov_real={ov_real:.3f}"
            }

        # DF harus cukup bagus juga, baru dianggap real
        if df_score >= DF_PASS:
            return {
                "is_live": True,
                "reason": "ok",
                "prob_real": ov_real,
                "prob_spoof": ov_spoof,
                "df_is_real": df_is_real,
                "df_score": df_score,
                "model": "anti-spoof-mn3(+deepface)",
                "debug": f"PASS(ov_strong_live) ov_real={ov_real:.3f} df_is_real={df_is_real} df_score={df_score:.3f}"
            }

        # Di sini: ov_real tinggi tapi DF score rendah -> treat as spoof
        return {
            "is_live": False,
            "reason": "spoof_detected",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3+deepface",
            "debug": f"FAIL(ov_strong_live_df_low) ov_real={ov_real:.3f} df_score={df_score:.3f}"
        }

    # --- Pass kalau tidak masuk blok strong-live di atas
    if df_score is None:
        # No DF module available -> rely on OpenVINO only
        if ov_real >= OV_PASS:
            return {
                "is_live": True,
                "reason": "ok",
                "prob_real": ov_real,
                "prob_spoof": ov_spoof,
                "df_is_real": None,
                "df_score": None,
                "model": "anti-spoof-mn3",
                "debug": f"PASS(ov_only) ov_real={ov_real:.3f}"
            }
        # uncertain band
        return {
            "is_live": False,
            "reason": "liveness_uncertain",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": None,
            "df_score": None,
            "model": "anti-spoof-mn3",
            "debug": f"UNCERTAIN(ov_only) ov_real={ov_real:.3f}"
        }

    # DF available: kombinasi keduanya di area normal
    if (ov_real >= OV_PASS and df_score >= DF_PASS):
        return {
            "is_live": True,
            "reason": "ok",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3+deepface",
            "debug": f"PASS(both) ov_real={ov_real:.3f} df_score={df_score:.3f}"
        }

    # allow DF super-strong to rescue weak webcam, but still not too low OV
    if (df_score >= DF_STRONG_LIVE and ov_real >= OV_UNCERTAIN_LOW):
        return {
            "is_live": True,
            "reason": "ok",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3+deepface",
            "debug": f"PASS(df_rescue) ov_real={ov_real:.3f} df_score={df_score:.3f}"
        }

    # otherwise -> uncertain / spoof ringan (blokir juga)
    return {
        "is_live": False,
        "reason": "liveness_uncertain" if ov_real >= OV_UNCERTAIN_LOW else "spoof_detected",
        "prob_real": ov_real,
        "prob_spoof": ov_spoof,
        "df_is_real": df_is_real,
        "df_score": df_score,
        "model": "anti-spoof-mn3+deepface",
        "debug": f"FAIL/UNCERTAIN ov_real={ov_real:.3f} df_is_real={df_is_real} df_score={df_score}"
    }


# ================== ARCFace Embedding ==================

def get_arcface_embedding(image_bytes: bytes) -> np.ndarray:
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


# ================== MATCH CORE (NO LIVENESS) ==================

def _recognize_no_liveness(image_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    db = load_db()
    if not db:
        return {"match": False, "reason": "Database kosong."}

    threshold = float(threshold)
    query_emb = get_arcface_embedding(image_bytes)

    best_id: Optional[str] = None
    best_name: Optional[str] = None
    best_dist: float = 999.0

    second_id: Optional[str] = None
    second_dist: float = 999.0

    # hitung jarak terbaik PER ORANG (min dari semua embedding milik orang itu)
    for emp_id, data in db.items():
        name = data.get("name")
        embs = data.get("embeddings", []) or []
        if not embs:
            continue

        emp_best = 999.0
        for emb in embs:
            emb_vec = np.array(emb, dtype=np.float32)
            dist = float(np.linalg.norm(query_emb - emb_vec))
            if dist < emp_best:
                emp_best = dist

        if emp_best < best_dist:
            second_id, second_dist = best_id, best_dist
            best_id, best_dist = emp_id, emp_best
            best_name = name
        elif emp_best < second_dist:
            second_id, second_dist = emp_id, emp_best

    if best_id is None:
        return {"match": False, "reason": "Tidak ada wajah cocok.", "threshold": threshold}

    margin = float(second_dist - best_dist)
    margin_ok = (second_id is None) or (margin >= MATCH_MARGIN) or (second_dist >= 998.0)

    if best_dist < threshold and margin_ok:
        confidence = "high" if best_dist < 2.0 else ("medium" if best_dist < 3.0 else "low")
        return {
            "match": True,
            "employee_id": best_id,
            "name": best_name,
            "distance": float(best_dist),
            "threshold": threshold,
            "confidence": confidence,
            # debug tambahan (boleh dipakai/tidak)
            "second_best_employee_id": second_id,
            "second_best_distance": float(second_dist),
            "margin": margin,
            "margin_ok": margin_ok,
        }

    return {
        "match": False,
        "reason": "Tidak ada wajah cocok.",
        "best_distance": float(best_dist),
        "threshold": threshold,
        # debug tambahan
        "second_best_employee_id": second_id,
        "second_best_distance": float(second_dist),
        "margin": margin,
        "margin_ok": margin_ok,
    }


# ================== REGISTER ==================

def register_face(employee_id: str, name: str, image_bytes: bytes) -> Dict[str, Any]:
    if ENROLL_REQUIRE_LIVENESS:
        live = detect_spoof(image_bytes)
        if not live["is_live"]:
            raise ValueError(f"Enrollment ditolak ({live['reason']}): {live.get('debug')}")

    db = load_db()
    emb = get_arcface_embedding(image_bytes)
    emb_list = emb.tolist()

    if employee_id in db:
        db[employee_id]["embeddings"].append(emb_list)
        db[employee_id]["name"] = name
    else:
        db[employee_id] = {"name": name, "embeddings": [emb_list]}

    save_db(db)
    return {"employee_id": employee_id, "name": name, "num_embeddings": len(db[employee_id]["embeddings"])}


# ================== RECOGNIZE (SINGLE) ==================

def recognize_face(image_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    live = detect_spoof(image_bytes)

    if not live["is_live"]:
        return {
            "match": False,
            "liveness_ok": False,
            "liveness_reason": live.get("reason"),
            "liveness_prob_real": live.get("prob_real"),
            "liveness_prob_spoof": live.get("prob_spoof"),
            "liveness_df_is_real": live.get("df_is_real"),
            "liveness_df_score": live.get("df_score"),
            "liveness_model": live.get("model"),
            "reason": live.get("reason", "spoof_detected"),
            "debug_liveness": live.get("debug"),
        }

    rec = _recognize_no_liveness(image_bytes, threshold=threshold)
    rec.update({
        "liveness_ok": True,
        "liveness_reason": live.get("reason"),
        "liveness_prob_real": live.get("prob_real"),
        "liveness_prob_spoof": live.get("prob_spoof"),
        "liveness_df_is_real": live.get("df_is_real"),
        "liveness_df_score": live.get("df_score"),
        "liveness_model": live.get("model"),
        "debug_liveness": live.get("debug"),
    })
    return rec


# ================== RECOGNIZE (MULTI) ==================

def recognize_face_multi(images_bytes: List[bytes], threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Vote across N frames:
    - If >= ceil(N*0.6) frames pass liveness => accept, then recognize using best liveness frame.
    - Else if any strong spoof => spoof_detected
    - Else => liveness_uncertain
    """
    if not images_bytes:
        return {"match": False, "liveness_ok": False, "reason": "quality_low", "debug_liveness": "no images"}

    results = [detect_spoof(b) for b in images_bytes]

    pass_count = sum(1 for r in results if r.get("is_live"))
    n = len(results)
    need = int(np.ceil(n * 0.6))

    # pick best frame for recognition (highest ov real; None -> 0.0)
    best_idx = int(np.argmax([_safe_float(r.get("prob_real"), 0.0) for r in results]))
    best_live = results[best_idx]

    if pass_count >= need:
        rec = _recognize_no_liveness(images_bytes[best_idx], threshold=threshold)
        rec.update({
            "liveness_ok": True,
            "liveness_reason": "ok",
            "liveness_prob_real": best_live.get("prob_real"),
            "liveness_prob_spoof": best_live.get("prob_spoof"),
            "liveness_df_is_real": best_live.get("df_is_real"),
            "liveness_df_score": best_live.get("df_score"),
            "liveness_model": best_live.get("model"),
            "debug_liveness": f"VOTE pass={pass_count}/{n}, best_idx={best_idx}, best={best_live.get('debug')}",
            "frames": results,
        })
        return rec

    if any(r.get("reason") == "spoof_detected" for r in results):
        return {
            "match": False,
            "liveness_ok": False,
            "reason": "spoof_detected",
            "debug_liveness": f"VOTE fail pass={pass_count}/{n}",
            "frames": results,
        }

    return {
        "match": False,
        "liveness_ok": False,
        "reason": "liveness_uncertain",
        "debug_liveness": f"VOTE uncertain pass={pass_count}/{n}",
        "frames": results,
    }


# ================== DELETE USER ==================

def delete_user(employee_id: str) -> Dict[str, Any]:
    db = load_db()
    if employee_id not in db:
        return {"deleted": False, "reason": "User tidak ditemukan"}
    del db[employee_id]
    save_db(db)
    return {"deleted": True, "employee_id": employee_id}
