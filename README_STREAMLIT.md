# Streamlit Experiment Dashboard

Dashboard ini menambahkan UI interaktif untuk eksperimen thesis prompt engineering pada HumanEval tanpa mengubah notebook yang sudah ada.

## Jalankan

Disarankan memakai Python environment yang sama dengan notebook:

```powershell
.\run_streamlit.ps1
```

Jika muncul `.env tidak lengkap atau rusak`, berarti folder `.env` di repo tidak berisi virtual environment lengkap. Dalam kasus itu buka terminal dari environment yang dipakai Jupyter notebook, lalu jalankan:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Atau manual:

```powershell
.\.env\Scripts\python.exe -m pip install -r requirements.txt
.\.env\Scripts\python.exe -m streamlit run streamlit_app.py
```

Jangan pakai `streamlit run streamlit_app.py` jika command itu mengarah ke Python global yang berbeda dari kernel notebook. Gejalanya biasanya CUDA/model/package tidak terbaca padahal notebook bisa jalan.

## Fitur

- Pilih model Hugging Face, termasuk `deepseek-ai/deepseek-coder-6.7b-instruct`.
- Edit prompt template untuk `zero_shot`, `few_shot`, `cot`, dan `hint`.
- Pilih mode `Fixed Strategy` atau `Online Bandit`.
- Atur jumlah soal, repeat, max token, sampling, temperature, seed, timeout, alpha, dan force explore.
- Jalankan generation dan evaluasi langsung dari Streamlit.
- Lihat `Pass@1`, average reward, compile rate, distribusi strategi, tabel hasil, dan ekspor CSV/JSONL.
- Status run lebih jelas: model loading, start time, progress generation, live Pass@1, compile rate, reward, dan log task terakhir.

## Catatan

- Mulai dari 3 sampai 5 soal dulu karena model 6.7B berat.
- Evaluasi menjalankan kode hasil model dengan timeout seperti notebook, jadi jangan gunakan pada kode yang tidak dipercaya di luar eksperimen lokal.
- Notebook asli tidak diubah.
