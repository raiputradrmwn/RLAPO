$ErrorActionPreference = "SilentlyContinue"

$matches = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -and
        ($_.CommandLine -like "*streamlit_app.py*" -or $_.CommandLine -like "*streamlit*run*")
    }

if (-not $matches) {
    Write-Host "No Streamlit dashboard process found."
    exit 0
}

foreach ($proc in $matches) {
    Write-Host "Stopping PID $($proc.ProcessId): $($proc.Name)"
    Stop-Process -Id $proc.ProcessId -Force
}

Write-Host "Streamlit dashboard processes stopped."
