$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".env\Scripts\python.exe"
$Pyvenv = Join-Path $ProjectRoot ".env\pyvenv.cfg"

if ((Test-Path -LiteralPath $Python) -and (Test-Path -LiteralPath $Pyvenv)) {
    & $Python -m streamlit run (Join-Path $ProjectRoot "streamlit_app.py")
    exit $LASTEXITCODE
}

Write-Warning ".env tidak lengkap atau rusak. Fallback ke Python aktif di PATH. Jika notebook memakai environment lain, jalankan command ini dari terminal environment notebook."
python -m streamlit run (Join-Path $ProjectRoot "streamlit_app.py")
