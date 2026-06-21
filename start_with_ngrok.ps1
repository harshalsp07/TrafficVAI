# TrafficAI - Start with Ngrok
# This script starts the backend, frontend, and ngrok tunnel
# Requires: ngrok installed and NGROK_AUTHTOKEN set

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TrafficAI - Starting with Ngrok Tunnel" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if ngrok authtoken is set
if (-not $env:NGROK_AUTHTOKEN) {
    Write-Host "[!] NGROK_AUTHTOKEN not set." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Setup instructions:" -ForegroundColor White
    Write-Host "1. Go to https://dashboard.ngrok.com" -ForegroundColor Gray
    Write-Host "2. Click 'Your Authtoken' in the left sidebar" -ForegroundColor Gray
    Write-Host "3. Copy the token" -ForegroundColor Gray
    Write-Host "4. Run: `$env:NGROK_AUTHTOKEN='your_token_here'" -ForegroundColor Gray
    Write-Host "   Or add it to the .env file" -ForegroundColor Gray
    Write-Host ""
    
    $token = Read-Host "Enter your ngrok authtoken (or press Enter to skip ngrok)"
    if ($token) {
        $env:NGROK_AUTHTOKEN = $token
    } else {
        Write-Host "Starting without ngrok..." -ForegroundColor Yellow
    }
}

# Start Backend
Write-Host "[1/3] Starting FastAPI backend on port 8000..." -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    Set-Location "C:\work\ML_Models\Trafic\backend"
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 2>&1
}

# Wait for backend to be ready
Write-Host "     Waiting for backend to start..." -ForegroundColor Gray
$retries = 0
$backendReady = $false
while ($retries -lt 15 -and -not $backendReady) {
    Start-Sleep -Seconds 1
    $retries++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "     Backend is ready!" -ForegroundColor Green
        }
    } catch {
        # Backend not ready yet
    }
}

if (-not $backendReady) {
    Write-Host "[!] Backend may still be starting. Continuing anyway..." -ForegroundColor Yellow
}

# Start Frontend
Write-Host "[2/3] Starting React frontend on port 5173..." -ForegroundColor Green
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "C:\work\ML_Models\Trafic\frontend"
    npm run dev 2>&1
}

Start-Sleep -Seconds 2

# Start Ngrok
if ($env:NGROK_AUTHTOKEN) {
    Write-Host "[3/3] Starting ngrok tunnel to port 8000..." -ForegroundColor Green
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  Ngrok URLs will appear below:" -ForegroundColor Cyan
    Share-Org = "https://dashboard.ngrok.com" -ForegroundColor Cyan
    Write-Host "  Local:  http://localhost:8000" -ForegroundColor White
    Write-Host "  Ngrok:  Check the URL below" -ForegroundColor White
    Write-Host "  Web UI: http://localhost:4040" -ForegroundColor White
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    
    ngrok http 8000 --log=stdout
} else {
    Write-Host "[3/3] Skipping ngrok (authtoken not set)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  Local Access Only:" -ForegroundColor Cyan
    Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
    Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor Gray
    
    # Keep script running
    try {
        while ($true) { Start-Sleep -Seconds 5 }
    } finally {
        Write-Host "Shutting down..." -ForegroundColor Yellow
        Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
        Stop-Job -Job $frontendJob -ErrorAction SilentlyContinue
    }
}
