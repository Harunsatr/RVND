# MFVRPTW Route Optimization

Dokumentasi proyek optimasi distribusi obat dengan pendekatan **Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW)**. Seluruh angka dan parameter mengikuti dokumen referensi (DOCX) yang diberikan.

## Latar Belakang
- Gudang mendistribusikan obat ke 10 pelanggan (rumah sakit, klinik, puskesmas).
- Armada heterogen: tipe A (kapasitas 60, 2 unit), B (100, 2 unit), C (150, 1 unit).
- Setiap pelanggan memiliki permintaan, service time, serta time window.
- Tujuan: meminimalkan kombinasi jarak tempuh, waktu perjalanan + layanan, dan keterlambatan (soft time window).

## Alur Metode
1. **Distance & Time Matrix**: Jarak Euclidean dari koordinat, dengan asumsi 1 km = 1 menit.
2. **Sweep Algorithm**: Mengurutkan pelanggan berdasarkan sudut polar dan membentuk cluster sesuai kapasitas (1 cluster = 1 kendaraan, memperhatikan ketersediaan unit).
3. **Nearest Neighbor (NN)**: Membuat rute awal per cluster dari depot.
4. **Ant Colony System (ACS)**: Optimasi rute per cluster dengan parameter DOCX (m=2, α=1, β=2, ρ=0,2, q₀=0,85, iterasi=2).
5. **RVND**: Randomized Variable Neighborhood Descent (2-opt, swap, relocate) untuk penyempurnaan rute.
6. **Final Integration**: Menggabungkan hasil, menghitung metrik total, dan menyiapkan artefak final.
7. **Streamlit GUI**: Menyajikan hasil secara interaktif tanpa perhitungan ulang.

## Struktur Proyek
```
Program/
├─ data/processed/        # Artefak hasil parsing & optimasi (JSON)
├─ docs/                  # Dokumentasi ringkas & dokumentasi_id.md
├─ gui/                   # Aplikasi Streamlit (app.py)
├─ acs_solver.py          # Implementasi ACS per cluster
├─ distance_time.py       # Matriks jarak & waktu
├─ rvnd.py                # Tahap RVND
├─ sweep_nn.py            # Sweep + inisialisasi NN
└─ final_integration.py   # Integrasi pipeline & final_solution
```
Artefak final utama:
- `Program/data/processed/final_solution.json`
- `Program/docs/final_summary.md`
- `Program/docs/dokumentasi_id.md`

## Menjalankan Pipeline (Opsional)
Semua hasil sudah dibekukan. Langkah berikut hanya diperlukan jika ingin regenerasi:
```powershell
cd "E:/Kerja Remote/Jokian/Joki Matematika (exe)/Program"
..\.venv\Scripts\python.exe distance_time.py
..\.venv\Scripts\python.exe sweep_nn.py
..\.venv\Scripts\python.exe acs_solver.py
..\.venv\Scripts\python.exe rvnd.py
..\.venv\Scripts\python.exe final_integration.py
```

## Menjalankan Dashboard Streamlit
```powershell
cd "E:/Kerja Remote/Jokian/Joki Matematika (exe)"
.\.venv\Scripts\python.exe -m pip install streamlit plotly pandas
.\.venv\Scripts\streamlit.exe run Program\gui\app.py
```
Buka `http://localhost:8501` untuk melihat dashboard (KPI, plot rute, detail cluster, perbandingan ACS vs RVND, tombol unduh JSON/CSV/Markdown).

## Catatan
- Jangan ubah angka di artefak hasil (khususnya `final_solution.json`).
- GUI hanya membaca data; tidak ada optimasi ulang di sisi front-end.
- Dokumentasi rinci tersedia di `Program/docs/dokumentasi_id.md`.
