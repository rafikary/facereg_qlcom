
# **📌 README.md — QLCOM Face Recognition API**

Sistem face recognition untuk kebutuhan HRIS (absensi & verifikasi identitas) menggunakan FastAPI + ArcFace.
Mendukung multi-foto, multi-pose, kamera laptop/HP, masker, dan variasi sudut wajah.

---

## **1. Fitur Utama**

* Register wajah karyawan (multi-embedding per user)
* Face recognition menggunakan ArcFace (512-d vector)
* Confidence score: **high / medium / low**
* Mendukung kamera laptop/webcam/HP
* API siap dipakai frontend web / mobile
* Tes via Swagger atau HTML kamera
* Database sederhana menggunakan JSON

---

## **2. Struktur Folder**

```
project/
│── main.py             → FastAPI routing
│── engine.py           → Logic face recognition
│── faces_db.json       → Database embedding (auto-create)
│── requirements.txt
│── README.md
│── camera_test.html    → Cek wajah via kamera
│── enroll_camera.html  → Daftar wajah via kamera
└── venv/               → Virtual environment (tidak di-commit)
```

---

## **3. Cara Menjalankan (Run Locally)**

### **3.1. Buat & aktifkan virtual environment**

``` 
python -m venv venv
```

**Windows:**

```
venv\Scripts\activate
```

**Mac / Linux:**

```
source venv/bin/activate
```

---

### **3.2. Install dependencies**

```
pip install -r requirements.txt
```

---

### **3.3. Jalankan server**

```
uvicorn main:app --reload
```

Server aktif di:

* API Base URL → [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger UI → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## **4. Endpoint API**

### **4.1. GET /**

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

### **4.2. POST /register**

Mendaftarkan wajah karyawan.

**Form-data:**

* employee_id
* name
* image (file)

**Response:**

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

### **4.3. POST /recognize**

Mencocokkan wajah ke database.

**Form-data:**

* image

**Response (match):**

```json
{
  "status": "success",
  "data": {
    "match": true,
    "employee_id": "RAFIKA123",
    "name": "Rafika",
    "distance": 2.85,
    "threshold": 3.5,
    "confidence": "medium"
  }
}
```

**Response (no match):**

```json
{
  "status": "success",
  "data": {
    "match": false,
    "reason": "Tidak ada wajah cocok.",
    "best_distance": 6.01,
    "threshold": 3.5
  }
}
```

---

### **4.4. GET /users**

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

### **4.5. DELETE /user/{employee_id}**

Menghapus data wajah karyawan.

---

## **5. Integrasi ke HRIS**

Frontend hanya memakai:

* `POST /register` → daftar wajah
* `POST /recognize` → absensi / login wajah
* `GET /users` → list user
* `DELETE /user/{employee_id}` → hapus user

### Base URL lokal

```
http://127.0.0.1:8000
```

### Base URL setelah deploy

```
https://face.company.com
```

Frontend cukup mengganti **base URL**, endpoint tetap sama.

---

## **6. Catatan Deployment**

* Folder `venv/` jangan di-upload ke Git
* Developer cukup menjalankan:

```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

* `faces_db.json` otomatis dibuat saat user pertama registrasi
* Pastikan server production menggunakan VPS / server kantor
* Bisa ditambah API Key / Token auth (opsional)

---

## **7. Lisensi / Penggunaan**

Digunakan untuk kebutuhan internal HRIS QLCOM.
Library pihak ketiga mengikuti lisensi masing-masing.