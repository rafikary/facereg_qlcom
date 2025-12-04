
# QLCOM Face Recognition API

Face recognition untuk HRIS (absensi & verifikasi identitas) berbasis **FastAPI + ArcFace**.
Dilengkapi **liveness / anti-spoofing** untuk mengurangi risiko absensi menggunakan foto layar/galeri/print.

---

## Fitur

* Register wajah karyawan (multi-embedding per user / multi-pose).
* Face recognition menggunakan ArcFace (embedding 512 dimensi).
* Confidence: `high / medium / low`.
* Liveness / anti-spoofing (2 layer):

  * OpenVINO Runtime + ONNX: model `anti-spoof-mn3.onnx` → `prob_real` vs `prob_spoof`
  * DeepFace anti_spoofing: menghasilkan `is_real` dan `antispoof_score`
* Siap dipakai dari web/mobile.
* Testing via Swagger dan halaman HTML kamera.
* Penyimpanan embedding sederhana: `faces_db.json`.

> Catatan: bila modul anti_spoofing DeepFace tidak tersedia (dependency belum ada), sistem tetap bisa berjalan dengan OpenVINO sebagai liveness utama.

---

## Struktur Folder

```
project/
├─ main.py              # FastAPI routing
├─ engine.py            # face recognition + liveness logic
├─ faces_db.json        # database embedding (auto-create)
├─ requirements.txt
├─ README.md
├─ camera_test.html     # test kamera (recognize)
├─ enroll_camera.html   # enroll via kamera
└─ models/
   └─ anti-spoof-mn3.onnx
```

---

## Menjalankan Secara Lokal

### 1) Buat dan aktifkan virtual environment

```bash
python -m venv venv
```

Windows (PowerShell):

```powershell
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Jika ingin mengaktifkan anti_spoofing bawaan DeepFace (opsional, butuh PyTorch CPU):

```bash
pip install lightphe==0.0.15
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3) Siapkan model liveness (OpenVINO)

Pastikan file ini ada:

* `models/anti-spoof-mn3.onnx`

Contoh download PowerShell:

```powershell
mkdir models
$u="https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/anti-spoof-mn3/anti-spoof-mn3.onnx"
Invoke-WebRequest -Uri $u -OutFile "models/anti-spoof-mn3.onnx"
```

### 4) Jalankan server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Akses:

* Base URL: `http://127.0.0.1:8000`
* Swagger: `http://127.0.0.1:8000/docs`

---

## Endpoint

### GET `/`

Health check.

### POST `/register`

Mendaftarkan wajah karyawan (menambah embedding ke database).

Form-data:

* `employee_id`
* `name`
* `image` (file)

Contoh response:

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

### POST `/recognize`

Mencocokkan wajah dengan database.

Form-data:

* `image` (file)

Contoh response (liveness lolos + match):

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

Contoh response (liveness lolos + tidak match):

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

Contoh response (ditolak liveness):

```json
{
  "status": "success",
  "data": {
    "match": false,
    "liveness_ok": false,
    "reason": "spoof_detected",
    "liveness_model": "anti-spoof-mn3(+deepface)",
    "debug_liveness": "..."
  }
}
```

Reason yang umum:

* `spoof_detected` → terindikasi foto/layar/print
* `liveness_uncertain` → hasil belum konsisten, disarankan retake
* `quality_low` → kualitas wajah/crop buruk (gelap/blur/kecil)

### GET `/users`

Daftar user yang terdaftar.

### DELETE `/user/{employee_id}`

Hapus user dari database.

---

## Catatan Liveness / Anti-Spoofing

Liveness dijalankan **sebelum** face recognition. Jika liveness gagal, proses recognize tidak dilanjutkan.

Faktor yang mempengaruhi hasil:

* pencahayaan terlalu gelap/terang (backlight)
* blur akibat gerakan / autofocus
* kompresi gambar (mis. JPEG dari canvas)
* wajah terlalu kecil di frame

Saran:

* wajah cukup dekat dan jelas
* cahaya stabil
* tahan posisi 1–2 detik saat capture

---

## Deployment Singkat

* Folder `venv/` jangan di-commit.
* Jalankan production:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

* `faces_db.json` dibuat otomatis saat pendaftaran pertama.
* Untuk production, disarankan tambah API key / auth layer.
