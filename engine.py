import json
import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
from deepface import DeepFace
import openvino as ov

DB_PATH = "faces_db.json"
DETECTOR_BACKEND = "mtcnn"

# Liveness detection configuration
LIVENESS_MODEL_PATH = "models/anti-spoof-mn3.onnx"

# OpenVINO anti-spoof thresholds
OV_PASS = 0.90
OV_UNCERTAIN_LOW = 0.70
OV_STRONG_SPOOF = 0.50

# DeepFace anti-spoof thresholds
DF_PASS = 0.80
DF_STRONG_LIVE = 0.90
DF_STRONG_SPOOF = 0.30

# Quality thresholds
MIN_FACE_AREA = 90 * 90
MIN_FACE_CONF = 0.70
MIN_BLUR_VAR = 10.0
MIN_BLUR_VAR_REGISTER = 7.0
MIN_BRIGHT = 5.0
MAX_BRIGHT = 250.0

# Image size limits
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 720
JPEG_QUALITY_HIGH = 95
JPEG_QUALITY_NORMAL = 85

ENROLL_REQUIRE_LIVENESS = False

# Face matching configuration
DEFAULT_THRESHOLD = 3.6
MATCH_MARGIN = 0.20

# OpenVINO runtime objects
_OV_CORE: Optional[ov.Core] = None
_OV_COMPILED = None
_OV_OUTPUT = None


def load_liveness_model() -> None:
    """Initialize liveness detection models at startup."""
    global _OV_CORE, _OV_COMPILED, _OV_OUTPUT

    try:
        import torch
        print(f"[LIVENESS] PyTorch version: {torch.__version__}")
    except ImportError:
        print("[WARNING] PyTorch not installed - DeepFace anti-spoofing disabled")
        print("[WARNING] Install with: pip install torch torchvision")

    if not os.path.exists(LIVENESS_MODEL_PATH):
        print(f"[LIVENESS] ⚠️  OpenVINO model not found: {LIVENESS_MODEL_PATH}")
        print("[LIVENESS] Anti-spoof detection will be DISABLED (INSECURE)")
        _OV_CORE = None
        _OV_COMPILED = None
        _OV_OUTPUT = None
        return

    _OV_CORE = ov.Core()
    model = _OV_CORE.read_model(LIVENESS_MODEL_PATH)
    _OV_COMPILED = _OV_CORE.compile_model(model, "CPU")
    _OV_OUTPUT = _OV_COMPILED.output(0)
    print(f"[LIVENESS] ✅ OpenVINO model loaded: {LIVENESS_MODEL_PATH}")


# Database utilities

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db: Dict[str, Any]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


# Image processing utilities

def bytes_to_bgr_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan jpg/png/jpeg valid.")
    return img


def _resize_if_large(img: np.ndarray, max_width: int = MAX_IMAGE_WIDTH, max_height: int = MAX_IMAGE_HEIGHT) -> np.ndarray:
    """
    Resize image jika terlalu besar (untuk optimasi processing speed dan memory).
    Maintain aspect ratio.
    """
    h, w = img.shape[:2]
    
    # Kalau sudah di bawah limit, return as-is
    if w <= max_width and h <= max_height:
        return img
    
    # Hitung scaling factor (pilih yang lebih kecil)
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    # Resize dengan maintain aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"[RESIZE] Image resized from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")
    
    return resized


def _compress_image(img: np.ndarray, quality: int = JPEG_QUALITY_NORMAL, max_size_kb: int = 500) -> bytes:
    # Coba dengan quality awal
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    img_bytes = buffer.tobytes()
    size_kb = len(img_bytes) / 1024
    
    # Kalau masih terlalu besar, turunkan quality secara iteratif
    while size_kb > max_size_kb and quality > 60:
        quality -= 5
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        img_bytes = buffer.tobytes()
        size_kb = len(img_bytes) / 1024
    
    print(f"[COMPRESS] Image compressed to {size_kb:.1f} KB (quality={quality})")
    return img_bytes


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _expand_bbox(x: int, y: int, w: int, h: int, pct: int, W: int, H: int) -> Tuple[int, int, int, int]:
    dx = int((w * pct / 100.0) / 2.0)
    dy = int((h * pct / 100.0) / 2.0)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(W, x + w + dx)
    y2 = min(H, y + h + dy)
    return x1, y1, x2, y2


def _preprocess_ov(face_bgr_u8: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_bgr_u8, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Model-specific normalization parameters
    mean_bgr = np.array([107.8395, 119.5950, 151.2405], dtype=np.float32)  # B,G,R
    scale_bgr = np.array([55.0035, 56.4570, 63.0105], dtype=np.float32)    # B,G,R
    face = (face - mean_bgr) / scale_bgr

    inp = np.transpose(face, (2, 0, 1))[None, ...]  # (1,3,128,128)
    return inp


def _infer_ov(face_bgr_u8: np.ndarray) -> Tuple[float, float]:
    inp = _preprocess_ov(face_bgr_u8)
    out = _OV_COMPILED([inp])[_OV_OUTPUT]
    out = np.array(out).reshape(2)
    p0, p1 = float(out[0]), float(out[1])

    s = p0 + p1
    if not (0.9 <= s <= 1.1) or p0 < 0 or p1 < 0 or p0 > 2 or p1 > 2:
        ex = np.exp(out - np.max(out))
        sm = ex / (np.sum(ex) + 1e-9)
        p0, p1 = float(sm[0]), float(sm[1])

    return p0, p1


def _detect_screen_photo(face_bgr_u8: np.ndarray) -> Tuple[bool, float, str]:
    try:
        gray = cv2.cvtColor(face_bgr_u8, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Detect moiré pattern via FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Cari peak di frekuensi tinggi (khas grid pattern layar)
        center_y, center_x = h // 2, w // 2
        r_inner = min(h, w) // 8  # skip DC component
        r_outer = min(h, w) // 3
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), r_outer, 255, -1)
        cv2.circle(mask, (center_x, center_y), r_inner, 0, -1)
        
        high_freq = magnitude * (mask / 255.0)
        high_freq_mean = np.mean(high_freq)
        high_freq_std = np.std(high_freq)
        
        # Foto layar punya peak di frekuensi tinggi (moiré)
        moire_score = high_freq_std / (high_freq_mean + 1e-6)
        
        # 2. Color cast detection (layar biasanya ada blue/green cast)
        b, g, r = cv2.split(face_bgr_u8.astype(np.float32))
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        # Hitung color imbalance
        total = b_mean + g_mean + r_mean + 1e-6
        b_ratio = b_mean / total
        g_ratio = g_mean / total
        r_ratio = r_mean / total
        
        # Layar LCD biasanya punya slight blue cast
        color_imbalance = abs(b_ratio - 0.33) + abs(g_ratio - 0.33) + abs(r_ratio - 0.33)
        
        # 3. Edge sharpness (foto layar lebih soft karena double-capture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_var = laplacian.var()
        
        # Threshold detection - MULTI-FACTOR untuk akurasi tinggi
        is_screen = False
        confidence = 0.0
        reason = ""
        
        # KOMBINASI INDIKATOR (perlu minimal 2 dari 3 faktor)
        indicators = []
        
        # 1. Moiré pattern (threshold dinaikkan untuk kurangi false positive)
        if moire_score > 8.0:  # VERY strong moiré (clearly screen photo)
            indicators.append(("moire_strong", 1.0, f"moire={moire_score:.2f}"))
        elif moire_score > 5.0:  # Moderate moiré
            indicators.append(("moire_moderate", 0.7, f"moire={moire_score:.2f}"))
        
        # 2. Color cast (only if very obvious)
        if color_imbalance > 0.15:  # Strong color imbalance
            indicators.append(("color_cast", 0.8, f"imbalance={color_imbalance:.3f}"))
        
        # 3. Edge softness (foto layar lebih soft)
        if edge_var < 30:  # Very soft edges
            indicators.append(("soft_edges", 0.7, f"edge_var={edge_var:.1f}"))
        
        # Decision: need at least 2 indicators OR 1 very strong indicator
        if len(indicators) >= 2:
            is_screen = True
            confidence = sum(ind[1] for ind in indicators) / len(indicators)
            reason = " + ".join(ind[2] for ind in indicators)
        elif len(indicators) == 1 and indicators[0][1] >= 1.0:  # Very strong single indicator
            is_screen = True
            confidence = indicators[0][1]
            reason = indicators[0][2]
        
        return is_screen, confidence, reason
        
    except Exception as e:
        # Kalau gagal analisis, return False (default pass)
        return False, 0.0, f"analysis_failed: {str(e)}"


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


# Liveness detection

def detect_spoof(image_bytes: bytes, skip_screen_detector: bool = False, lenient_quality: bool = False) -> Dict[str, Any]:
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
        from deepface import DeepFace  # Lazy import
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
        # Fallback to OpenVINO only
        try:
            from deepface import DeepFace  # Lazy import
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



    # Apply CLAHE for lighting normalization
    face_enhanced = _enhance_lighting(face_u8)

    # Quality validation
    fa = (meta or {}).get("facial_area", {}) or {}
    x = int(fa.get("x", 0) or 0)
    y = int(fa.get("y", 0) or 0)
    w = int(fa.get("w", 0) or 0)
    h = int(fa.get("h", 0) or 0)
    conf = _safe_float((meta or {}).get("confidence", 1.0), 1.0)

    qm = _quality_metrics(face_enhanced)
    blur_var = qm["blur_var"]
    bright = qm["bright"]
    area = w * h

    # Gunakan threshold blur yang tepat (lenient untuk register, ketat untuk recognize)
    blur_threshold = MIN_BLUR_VAR_REGISTER if lenient_quality else MIN_BLUR_VAR
    
    low_conf = conf is not None and conf < MIN_FACE_CONF
    too_small = area < MIN_FACE_AREA
    too_blur = blur_var < blur_threshold
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
            "debug": f"quality_low conf={conf:.2f} area={w}x{h} blur={blur_var:.1f} (need>={blur_threshold:.1f}) bright={bright:.1f}"
        }

    # 3) OpenVINO multi-crop inference from bbox
    # Gunakan enhanced image untuk inference
    ov_scores = []
    ov_spoofs = []

    if w > 0 and h > 0:
        for pct in (0, 20, 40):
            x1, y1, x2, y2 = _expand_bbox(x, y, w, h, pct, W, H)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_enhanced = _enhance_lighting(crop.astype(np.uint8))
            p_real, p_spoof = _infer_ov(crop_enhanced)
            ov_scores.append(p_real)
            ov_spoofs.append(p_spoof)
    else:
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

    # Liveness decision logic
    
    # Strong spoof detection
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

    # DeepFace spoof flag with OpenVINO cross-validation
    if df_is_real is False and ov_real <= 0.75:
        return {
            "is_live": False,
            "reason": "spoof_detected",
            "prob_real": ov_real,
            "prob_spoof": ov_spoof,
            "df_is_real": df_is_real,
            "df_score": df_score,
            "model": "anti-spoof-mn3(+deepface)",
            "debug": f"FAIL(df_flag_spoof_ov_uncertain) ov_real={ov_real:.3f} df_is_real={df_is_real} df_score={df_score}"
        }

    # OpenVINO MUST be >= 0.90 (primary filter)
    if ov_real >= OV_PASS:
        # DeepFace validation (primary decision)
        if df_score is not None:
            # Rule 1: DF explicitly flags spoof → ALWAYS REJECT
            if df_is_real == False:
                return {
                    "is_live": False,
                    "reason": "spoof_detected",
                    "prob_real": ov_real,
                    "prob_spoof": ov_spoof,
                    "df_is_real": df_is_real,
                    "df_score": df_score,
                    "model": "anti-spoof-mn3+deepface",
                    "debug": f"FAIL(df_flag_spoof) ov_real={ov_real:.3f} df_is_real={df_is_real} df_score={df_score:.3f}"
                }
            
            # Rule 2: DF score >= DF_PASS (0.55) → PASS
            if df_score >= DF_PASS:
                return {
                    "is_live": True,
                    "reason": "ok",
                    "prob_real": ov_real,
                    "prob_spoof": ov_spoof,
                    "df_is_real": df_is_real,
                    "df_score": df_score,
                    "model": "anti-spoof-mn3+deepface",
                    "debug": f"PASS(both_pass) ov_real={ov_real:.3f} df_score={df_score:.3f}"
                }
            
            # Rule 3: DF flag True + score >= 0.45 → PASS
            elif df_is_real == True and df_score >= 0.45:
                return {
                    "is_live": True,
                    "reason": "ok",
                    "prob_real": ov_real,
                    "prob_spoof": ov_spoof,
                    "df_is_real": df_is_real,
                    "df_score": df_score,
                    "model": "anti-spoof-mn3+deepface",
                    "debug": f"PASS(df_flag_real) ov_real={ov_real:.3f} df_score={df_score:.3f} df_is_real=True"
                }
            
            # Rule 4: Else → REJECT (score too low or uncertain)
            else:
                return {
                    "is_live": False,
                    "reason": "spoof_detected",
                    "prob_real": ov_real,
                    "prob_spoof": ov_spoof,
                    "df_is_real": df_is_real,
                    "df_score": df_score,
                    "model": "anti-spoof-mn3+deepface",
                    "debug": f"FAIL(df_low_score) ov_real={ov_real:.3f} df_score={df_score:.3f} (need>={DF_PASS})"
                }
        
        # No DF available, trust OpenVINO alone
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

    # OpenVINO < 0.90 → REJECT (no rescue)
    return {
        "is_live": False,
        "reason": "spoof_detected" if ov_real < OV_UNCERTAIN_LOW else "liveness_uncertain",
        "prob_real": ov_real,
        "prob_spoof": ov_spoof,
        "df_is_real": df_is_real,
        "df_score": df_score,
        "model": "anti-spoof-mn3+deepface" if df_score is not None else "anti-spoof-mn3",
        "debug": f"FAIL(ov_low) ov_real={ov_real:.3f} df_score={df_score}"
    }


# ================== ARCFace Embedding ==================

def get_arcface_embedding(image_bytes: bytes) -> np.ndarray:
    from deepface import DeepFace  # Lazy import
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


# Face enrollment

def register_face(employee_id: str, name: str, image_bytes: bytes) -> Dict[str, Any]:
    # 1. Resize image jika terlalu besar (optimasi speed & memory)
    img = bytes_to_bgr_image(image_bytes)
    img_resized = _resize_if_large(img)
    
    # 2. Compress untuk optimasi file size (max 500KB)
    image_bytes = _compress_image(img_resized, quality=JPEG_QUALITY_HIGH, max_size_kb=500)
    
    if ENROLL_REQUIRE_LIVENESS:
        # Skip screen detector & use lenient quality untuk enrollment (device beragam)
        live = detect_spoof(image_bytes, skip_screen_detector=True, lenient_quality=True)
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


# Single-frame recognition

def recognize_face(image_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    # 1. Resize image jika terlalu besar (optimasi speed & memory)
    img = bytes_to_bgr_image(image_bytes)
    img_resized = _resize_if_large(img)
    
    # 2. Compress untuk optimasi file size
    image_bytes = _compress_image(img_resized, quality=JPEG_QUALITY_NORMAL, max_size_kb=400)
    
    # 3. Liveness detection (use STRICT blur threshold for recognize)
    live = detect_spoof(image_bytes, lenient_quality=False)

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


# Multi-frame recognition

def recognize_face_multi(images_bytes: List[bytes], threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
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


# User management

def delete_user(employee_id: str) -> Dict[str, Any]:
    db = load_db()
    if employee_id not in db:
        return {"deleted": False, "reason": "User tidak ditemukan"}
    del db[employee_id]
    save_db(db)
    return {"deleted": True, "employee_id": employee_id}
