$ErrorActionPreference = "SilentlyContinue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Test-Endpoint {
    param([string]$Url)
    try {
        return (Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 1).StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Endpoint "http://127.0.0.1:8000/health")) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "title Bank API && uv run python -m bank_api" -WorkingDirectory $root
}

if (-not (Test-Endpoint "http://127.0.0.1:5173/")) {
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "title Bank Web && node apps\web\server.mjs" -WorkingDirectory $root
}

$deadline = (Get-Date).AddSeconds(30)
do {
    Start-Sleep -Milliseconds 500
    $apiReady = Test-Endpoint "http://127.0.0.1:8000/ready"
    $webReady = Test-Endpoint "http://127.0.0.1:5173/"
} until (($apiReady -and $webReady) -or (Get-Date) -ge $deadline)

if ($apiReady -and $webReady) {
    Start-Process "http://127.0.0.1:5173"
} else {
    Write-Host "启动失败，请检查 Bank API 和 Bank Web 两个窗口。"
    Read-Host "按回车关闭"
}
