# MFVRPTW - Multi-Fleet Vehicle Routing Problem with Time Windows

Sistem optimasi rute untuk distribusi obat dari gudang ke berbagai pelanggan (rumah sakit, klinik, puskesmas) menggunakan berbagai jenis kendaraan dengan batasan kapasitas dan time windows.

## 📋 Deskripsi Program

Program ini menyelesaikan masalah **Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW)** - sebuah masalah optimasi yang kompleks untuk menemukan rute distribusi paling efisien dengan:

- **Multi-Fleet**: Menggunakan berbagai jenis kendaraan (Motor, Mobil Kecil, Mobil Besar) dengan kapasitas dan biaya berbeda
- **Time Windows**: Setiap pelanggan memiliki waktu layanan yang harus dipenuhi
- **Kapasitas**: Setiap kendaraan memiliki batasan kapasitas maksimal
- **Optimasi Biaya**: Meminimalkan biaya tetap (fixed cost) dan biaya variabel (per km)

### 🎯 Pipeline Optimasi

Program ini menggunakan algoritma multi-tahap untuk menghasilkan solusi optimal:

1. **Perhitungan Matriks Jarak & Waktu** (`distance_time.py`)
   - Menghitung jarak Euclidean dari koordinat
   - Waktu tempuh: 1 km = 1 menit

2. **Sweep Algorithm** (`sweep_nn.py`)
   - Mengurutkan pelanggan berdasarkan sudut polar
   - Membentuk cluster berdasarkan kapasitas kendaraan
   - 1 cluster = 1 kendaraan

3. **Nearest Neighbor (NN)** (`sweep_nn.py`)
   - Inisialisasi rute awal untuk setiap cluster
   - Membuat urutan kunjungan pelanggan

4. **Ant Colony System (ACS)** (`acs_solver.py`)
   - Optimasi rute per cluster
   - Parameter: m=2, α=1, β=2, ρ=0.2, q₀=0.85, iterasi=2

5. **RVND (Randomized Variable Neighborhood Descent)** (`rvnd.py`)
   - Perbaikan akhir dengan operator:
     - 2-opt: Membalikkan segmen rute
     - Swap: Menukar posisi dua pelanggan
     - Relocate: Memindahkan pelanggan ke posisi lain

6. **Final Integration** (`final_integration.py`)
   - Menggabungkan semua hasil
   - Validasi solusi
   - Menghasilkan laporan final

### 📊 Dashboard Interaktif

Program dilengkapi dengan GUI berbasis **Streamlit** yang menampilkan:
- Input data pelanggan dan koordinat
- Visualisasi rute di peta
- Tabel detail rute per kendaraan
- Grafik perbandingan biaya
- Statistik dan metrik optimasi

## 🚀 Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/Harunsatr/RVND.git
cd RVND
```

2. Buat virtual environment (opsional tapi direkomendasikan):
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows CMD
python -m venv .venv
.venv\Scripts\activate.bat

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Jika file `requirements.txt` tidak ada, install manual:
```bash
pip install streamlit plotly pandas numpy
```

## 💻 Cara Menjalankan Program

### Opsi 1: Menjalankan Dashboard (Recommended)

Dashboard sudah berisi hasil optimasi yang telah dihitung. Cukup jalankan:

```bash
# Dari root folder project
streamlit run gui/app.py
```

Atau jika Anda menggunakan virtual environment:

```bash
# Windows
.\.venv\Scripts\streamlit.exe run gui\app.py

# Linux/Mac
.venv/bin/streamlit run gui/app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

### Opsi 2: Menjalankan Pipeline Optimasi (Opsional)

Jika Anda ingin menghitung ulang optimasi dari awal:

```bash
# Jalankan script secara berurutan
python distance_time.py
python sweep_nn.py
python acs_solver.py
python rvnd.py
python final_integration.py
```

⚠️ **Catatan**: Hasil optimasi sudah tersimpan di folder `data/processed/`. Anda hanya perlu menjalankan pipeline ini jika ingin mengubah data input atau parameter algoritma.

## 📁 Struktur Folder

```
Program/
├── data/
│   └── processed/              # Data hasil optimasi
│       ├── parsed_instance.json    # Data instance (depot, pelanggan, fleet)
│       ├── parsed_distance.json    # Matriks jarak dan waktu
│       ├── clusters.json           # Hasil clustering
│       ├── initial_routes.json     # Rute awal dari NN
│       ├── acs_routes.json         # Rute setelah ACS
│       ├── rvnd_routes.json        # Rute setelah RVND
│       └── final_solution.json     # Solusi akhir lengkap
│
├── docs/                       # Dokumentasi
│   ├── dokumentasi_id.md           # Dokumentasi lengkap (Bahasa Indonesia)
│   └── final_summary.md            # Ringkasan hasil optimasi
│
├── gui/                        # Aplikasi Streamlit
│   ├── app.py                      # File utama dashboard
│   ├── agents.py                   # Background agents
│   ├── components/                 # Komponen UI
│   └── tabs/                       # Tab-tab dashboard
│       ├── input_titik.py              # Tab input koordinat
│       ├── input_data.py               # Tab input data
│       ├── hasil.py                    # Tab hasil optimasi
│       └── graph_hasil.py              # Tab visualisasi grafik
│
├── distance_time.py            # Modul perhitungan jarak & waktu
├── sweep_nn.py                 # Sweep Algorithm + Nearest Neighbor
├── acs_solver.py               # Ant Colony System
├── rvnd.py                     # RVND Optimization
├── final_integration.py        # Integrasi dan validasi final
└── README.md                   # File ini
```

## 📖 Fitur Dashboard

### 1️⃣ Input Titik
- Input koordinat depot dan pelanggan
- Validasi data input
- Edit koordinat secara interaktif

### 2️⃣ Input Data
- Input data pelanggan (demand, time windows)
- Input data fleet (kapasitas, biaya)
- Konfigurasi parameter optimasi

### 3️⃣ Hasil Optimasi
- Tabel detail rute per kendaraan
- Total biaya (fixed + variable)
- Jumlah kendaraan yang digunakan
- Validasi constraint (kapasitas, time windows)

### 4️⃣ Visualisasi Grafik
- Peta rute dengan Plotly
- Grafik perbandingan biaya
- Timeline kunjungan pelanggan
- Statistik penggunaan kendaraan

## 🔧 Konfigurasi

### Parameter Algoritma

Parameter default dapat diubah di masing-masing file:

**ACS Parameters** (`acs_solver.py`):
```python
m = 2          # Jumlah semut
alpha = 1      # Pengaruh pheromone
beta = 2       # Pengaruh heuristic (jarak)
rho = 0.2      # Evaporation rate
q0 = 0.85      # Exploitation vs exploration
iterations = 2 # Jumlah iterasi
```

**RVND Parameters** (`rvnd.py`):
```python
max_iterations = 100  # Maksimal iterasi tanpa improvement
```

### Data Input

Data instance berada di `data/processed/parsed_instance.json` dengan struktur:
```json
{
  "depot": {
    "id": "depot",
    "x": 0,
    "y": 0
  },
  "customers": [
    {
      "id": "C1",
      "x": 10,
      "y": 20,
      "demand": 50,
      "ready_time": 0,
      "due_time": 480,
      "service_time": 15
    }
  ],
  "fleet": [
    {
      "id": "motor",
      "capacity": 100,
      "fixed_cost": 50000,
      "variable_cost_per_km": 2000
    }
  ]
}
```

## 📊 Hasil Optimasi

Hasil akhir tersimpan di `data/processed/final_solution.json` berisi:
- Rute untuk setiap kendaraan
- Total jarak dan waktu tempuh
- Biaya total (fixed + variable)
- Validasi semua constraint
- Timeline kunjungan

Contoh output rute:
```json
{
  "vehicle_type": "mobil_kecil",
  "route": ["depot", "C1", "C3", "C5", "depot"],
  "total_distance": 45.2,
  "total_cost": 140400,
  "customers_served": 3
}
```

## 🧪 Testing & Validasi

Program melakukan validasi otomatis terhadap:
- ✅ Semua pelanggan terlayani
- ✅ Kapasitas kendaraan tidak melebihi batas
- ✅ Time windows dipenuhi
- ✅ Setiap rute dimulai dan berakhir di depot
- ✅ Matriks jarak simetris
- ✅ Tidak ada jarak negatif

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:
1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Project ini dibuat untuk keperluan akademis/pembelajaran.

## 👨‍💻 Author

- **Harunsatr** - [GitHub](https://github.com/Harunsatr)

## 📚 Referensi

- Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning approach to the traveling salesman problem.
- Hansen, P., & Mladenović, N. (2001). Variable neighborhood search: Principles and applications.
- Gillett, B. E., & Miller, L. R. (1974). A heuristic algorithm for the vehicle-dispatch problem.

## ❓ FAQ

**Q: Program tidak bisa dijalankan, muncul error module not found?**
A: Pastikan semua dependencies sudah terinstall dengan `pip install -r requirements.txt`

**Q: Dashboard tidak menampilkan data?**
A: Pastikan file-file JSON di folder `data/processed/` ada dan tidak corrupt. Jika perlu, jalankan ulang pipeline optimasi.

**Q: Bagaimana cara mengubah data pelanggan?**
A: Edit file `data/processed/parsed_instance.json` kemudian jalankan ulang pipeline optimasi.

**Q: Apakah bisa digunakan untuk data yang lebih besar?**
A: Ya, program dapat di-scale untuk lebih banyak pelanggan, tetapi perlu penyesuaian parameter algoritma dan waktu komputasi akan lebih lama.

## 📞 Support

Jika ada pertanyaan atau masalah, silakan buat issue di [GitHub Issues](https://github.com/Harunsatr/RVND/issues)

---

⭐ Jika project ini membantu, jangan lupa berikan star di GitHub!
