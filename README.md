```md
# 📌 README.md — QLCOM Face Recognition API

Sistem face recognition untuk kebutuhan HRIS (absensi & verifikasi identitas) menggunakan FastAPI + ArcFace.  
Terdapat **lapisan keamanan liveness (anti-spoofing)** untuk mengurangi risiko absensi menggunakan foto layar/galeri/print.

---

## 1. Fitur Utama

- Register wajah karyawan (**multi-embedding per user / multi-pose**)
- Face recognition menggunakan **ArcFace** (vektor 512 dimensi)
- Confidence level: **high / medium / low**
- **Liveness / Anti-spoofing (Double Check / 2 Layer)**:
  - **Layer 1 (OpenVINO Runtime + ONNX)**: menjalankan model pretrained **`anti-spoof-mn3.onnx`** (Open Model Zoo) untuk menghasilkan `prob_real` vs `prob_spoof`
  - **Layer 2 (DeepFace built-in anti_spoofing)**: menggunakan model Silent-Face Anti-Spoofing (MiniFASNet family yang otomatis diunduh oleh DeepFace) untuk mengeluarkan `is_real` dan `antispoof_score`
  - Keputusan akhir: **HARUS lolos dua-duanya**. Jika salah satu gagal/berbeda → **spoof / liveness_uncertain** (disarankan retake).
- Mendukung kamera laptop/webcam/HP
- API dapat dipakai frontend web / mobile
- Tes via Swagger atau HTML kamera
- Database embedding menggunakan file JSON (`faces_db.json`)

---

## 2. Struktur Folder

```

project/
│── main.py             → FastAPI routing
│── engine.py           → Logic face recognition + liveness
│── faces_db.json       → Database embedding (auto-create)
│── requirements.txt
│── README.md
│── camera_test.html    → Cek wajah via kamera
│── enroll_camera.html  → Daftar wajah via kamera
│── models/
│   └── anti-spoof-mn3.onnx
└── venv/               → Virtual environment (tidak di-commit)

````

---

## 3. Cara Menjalankan (Run Locally)

### 3.1 Buat & aktifkan virtual environment

```bash
python -m venv venv
````

Windows:

```powershell
venv\Scripts\activate
```

Mac / Linux:

```bash
source venv/bin/activate
```

---

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

Catatan:

* DeepFace bisa membutuhkan dependency tambahan tertentu (mis. `lightphe`) tergantung versi.
* Jika memakai fitur anti-spoofing bawaan DeepFace, dibutuhkan PyTorch (CPU).

Contoh (opsional jika diperlukan):

```bash
pip install lightphe==0.0.15
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### 3.3 Siapkan model liveness (anti-spoof)

Pastikan model ONNX tersedia:

* File: `models/anti-spoof-mn3.onnx`

Contoh download via PowerShell:

```powershell
mkdir models
$u="https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/anti-spoof-mn3/anti-spoof-mn3.onnx"
Invoke-WebRequest -Uri $u -OutFile "models/anti-spoof-mn3.onnx"
```

---

### 3.4 Jalankan server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Server aktif di:

* API Base URL → [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger UI → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4. Endpoint API

### 4.1 GET `/`

Cek status API.

Contoh response:

```json
{
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
```

---

### 4.2 POST `/register`

Mendaftarkan wajah karyawan (menambah embedding ke database).

Form-data:

* `employee_id`
* `name`
* `image` (file)

Response:

```json
{
  "status": "success",
  "data": {
    "employee_id": "RAFIKA123",
    "name": "Rafika",
    "num_embeddings": 4
  }
}
```

---

### 4.3 POST `/recognize`

Mencocokkan wajah ke database.

Form-data:

* `image` (file)

#### Response (liveness lolos + match)

```json
{
  "status": "success",
  "data": {
    "match": true,
    "employee_id": "RAFIKA123",
    "name": "Rafika",
    "distance": 2.85,
    "threshold": 3.5,
    "confidence": "medium",
    "liveness_ok": true
  }
}
```

#### Response (liveness lolos + tidak match)

```json
{
  "status": "success",
  "data": {
    "match": false,
    "reason": "Tidak ada wajah cocok.",
    "best_distance": 6.01,
    "threshold": 3.5,
    "liveness_ok": true
  }
}
```

#### Response (ditolak karena spoof / liveness gagal)

```json
{
  "status": "success",
  "data": {
    "match": false,
    "liveness_ok": false,
    "reason": "spoof_detected",
    "liveness_model": "anti-spoof-mn3+deepface",
    "debug_liveness": "..."
  }
}
```

Keterangan `reason` yang umum:

* `spoof_detected` → terindikasi foto/layar/print
* `liveness_uncertain` → hasil dua model tidak konsisten (disarankan retake)
* `quality_low` → kualitas wajah/crop buruk (gelap/blur/kecil), disarankan ambil ulang

---

### 4.4 GET `/users`

Melihat daftar user terdaftar.

Contoh:

```json
{
  "status": "success",
  "data": [
    {
      "employee_id": "RAFIKA123",
      "name": "Rafika",
      "num_embeddings": 4
    }
  ]
}
```

---

### 4.5 DELETE `/user/{employee_id}`

Menghapus data user dari database.

---

## 5. Catatan Liveness / Anti-Spoofing

Liveness berjalan **sebelum** proses face recognition.
Jika liveness tidak lolos, proses recognize tidak dilanjutkan.

Hal yang mempengaruhi hasil:

* pencahayaan terlalu gelap/terlalu terang
* blur akibat gerakan / autofocus
* kompresi gambar (misalnya JPEG dari canvas)
* jarak wajah terlalu kecil di frame

Saran penggunaan:

* wajah cukup dekat (area wajah jelas)
* cahaya stabil dan tidak backlight
* tahan posisi 1–2 detik saat capture

---

## 6. Integrasi ke HRIS

Frontend hanya memakai:

* `POST /register` → pendaftaran wajah
* `POST /recognize` → verifikasi/absensi
* `GET /users` → list user
* `DELETE /user/{employee_id}` → hapus user

Base URL lokal:

```
http://127.0.0.1:8000
```

Base URL setelah deploy:

```
https://face.company.com
```

Endpoint tetap sama, hanya mengganti base URL.

---

## 7. Catatan Deployment

* Folder `venv/` jangan di-upload ke Git
* Jalankan server production:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

* File `faces_db.json` otomatis dibuat saat user pertama registrasi
* Disarankan tambah API Key / Token auth untuk environment production (opsional)

---

## 8. Lisensi / Penggunaan

Digunakan untuk kebutuhan internal HRIS QLCOM.
Library pihak ketiga mengikuti lisensi masing-masing.

```
::contentReference[oaicite:0]{index=0}
```
