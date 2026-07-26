[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "status", "open")]
    [string]$Action = "status",
    [switch]$Public
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localUrl = "http://127.0.0.1:8000"
$publicUrlFile = Join-Path $repoRoot "outputs\public-demo-url.txt"
$requiredModel = "gemma3:latest"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE."
    }
}

function Wait-For {
    param(
        [scriptblock]$Condition,
        [int]$TimeoutSeconds,
        [string]$FailureMessage,
        [int]$IntervalSeconds = 2
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (& $Condition) { return }
        Start-Sleep -Seconds $IntervalSeconds
    } while ((Get-Date) -lt $deadline)
    throw $FailureMessage
}

function Test-DockerEngine {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
    & docker info *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker CLI was not found. Install Docker Desktop and reopen the terminal."
    }
    if (Test-DockerEngine) { return }

    $candidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe")
    )
    $desktop = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $desktop) {
        throw "Docker engine is stopped and Docker Desktop was not found in a standard installation path."
    }
    Write-Step "Starting Docker Desktop"
    Start-Process -FilePath $desktop -WindowStyle Hidden
    Wait-For -TimeoutSeconds 120 -FailureMessage "Docker Desktop did not become ready within 120 seconds." -Condition {
        Test-DockerEngine
    }
}

function Get-OllamaCommand {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Ollama CLI was not found. Install Ollama and reopen the terminal."
    }
    return $command.Source
}

function Test-OllamaApi {
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

function Ensure-Ollama {
    $ollama = Get-OllamaCommand
    if (-not (Test-OllamaApi)) {
        Write-Step "Starting Ollama"
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
        Wait-For -TimeoutSeconds 60 -FailureMessage "Ollama did not become reachable within 60 seconds." -Condition {
            Test-OllamaApi
        }
    }
    $models = & $ollama list 2>$null
    if ($LASTEXITCODE -ne 0 -or -not ($models -match "(?m)^gemma3:latest\s")) {
        throw "Required model gemma3:latest is missing. Install it explicitly with: ollama pull gemma3:latest"
    }
}

function Get-JsonEndpoint {
    param([string]$Uri, [int]$TimeoutSeconds = 8)
    return Invoke-RestMethod -Uri $Uri -TimeoutSec $TimeoutSeconds
}

function Wait-WebReady {
    Write-Step "Waiting for the web container"
    Wait-For -TimeoutSeconds 180 -IntervalSeconds 3 -FailureMessage "The web container did not become healthy within 180 seconds." -Condition {
        $containerId = (& docker compose ps -q web 2>$null | Select-Object -First 1)
        if (-not $containerId) { return $false }
        $health = & docker inspect --format "{{.State.Health.Status}}" $containerId 2>$null
        return $LASTEXITCODE -eq 0 -and $health -eq "healthy"
    }
    $healthResult = Get-JsonEndpoint "$localUrl/healthz"
    $readyResult = Get-JsonEndpoint "$localUrl/readyz"
    if ($healthResult.status -ne "ok" -or -not $readyResult.ready) {
        throw "The local demo did not pass health and readiness checks."
    }
    return $readyResult
}

function Test-PublicUrlFormat {
    param([string]$Url)
    return $Url -match "^https://[a-z0-9-]+\.trycloudflare\.com/?$"
}

function Get-NewTunnelUrl {
    param([datetime]$Since)
    $sinceText = $Since.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $deadline = (Get-Date).AddSeconds(90)
    do {
        $logs = & docker compose logs --no-color --since $sinceText cloudflared 2>$null
        $match = [regex]::Match(($logs -join "`n"), "https://[a-z0-9-]+\.trycloudflare\.com")
        if ($match.Success) {
            $candidate = $match.Value
            try {
                $health = Get-JsonEndpoint "$candidate/healthz" 10
                $ready = Get-JsonEndpoint "$candidate/readyz" 10
                if ($health.status -eq "ok" -and $ready.ready) { return $candidate }
            } catch {
                # Quick Tunnel DNS can lag briefly behind URL allocation.
            }
        }
        Start-Sleep -Seconds 3
    } while ((Get-Date) -lt $deadline)
    throw "A new reachable Quick Tunnel URL was not available within 90 seconds."
}

function Save-And-OpenPublicUrl {
    param([string]$Url)
    if (-not (Test-PublicUrlFormat $Url)) {
        throw "Refusing to save or open an invalid Quick Tunnel URL."
    }
    $outputDirectory = Split-Path -Parent $publicUrlFile
    if (-not (Test-Path -LiteralPath $outputDirectory)) {
        New-Item -ItemType Directory -Path $outputDirectory | Out-Null
    }
    Set-Content -LiteralPath $publicUrlFile -Value $Url -Encoding ASCII
    Set-Clipboard -Value $Url
    Start-Process $Url
}

function Start-Demo {
    param([bool]$EnablePublic)
    Ensure-Docker
    Ensure-Ollama

    if ($EnablePublic) {
        Write-Step "Building and starting the public demo"
        Invoke-Checked docker @("compose", "--profile", "public", "up", "-d", "--build")
    } else {
        Write-Step "Building and starting the local demo"
        Invoke-Checked docker @("compose", "up", "-d", "--build", "web")
    }

    $ready = Wait-WebReady
    $publicUrl = $null
    if ($EnablePublic) {
        $tunnelStarted = (Get-Date).ToUniversalTime().AddSeconds(-2)
        Write-Step "Creating a fresh HTTP/2 Quick Tunnel"
        Invoke-Checked docker @(
            "compose", "--profile", "public", "up", "-d",
            "--force-recreate", "--no-deps", "cloudflared"
        )
        try {
            $publicUrl = Get-NewTunnelUrl -Since $tunnelStarted
            Save-And-OpenPublicUrl $publicUrl
        } catch {
            & docker compose stop cloudflared | Out-Host
            throw
        }
    } else {
        Start-Process $localUrl
    }

    Write-Host ""
    Write-Host "Demo ready" -ForegroundColor Green
    Write-Host "Local URL:      $localUrl"
    if ($publicUrl) { Write-Host "Public URL:     $publicUrl" }
    Write-Host "Web status:     healthy"
    Write-Host "Ollama status:  reachable ($requiredModel)"
    Write-Host "Indexed chunks: $($ready.indexed_chunks)"
    Write-Host "Stop command:   powershell -ExecutionPolicy Bypass -File scripts/demo.ps1 stop"
}

function Stop-Demo {
    Ensure-Docker
    Write-Step "Stopping only the demo services"
    Invoke-Checked docker @("compose", "--profile", "public", "stop")
    Write-Host "Demo services stopped. Volumes, images, indexes, and generated outputs were preserved."
}

function Show-Status {
    $dockerAvailable = [bool](Get-Command docker -ErrorAction SilentlyContinue)
    $dockerRunning = $dockerAvailable -and (Test-DockerEngine)
    $ollamaAvailable = [bool](Get-Command ollama -ErrorAction SilentlyContinue)
    $ollamaRunning = $ollamaAvailable -and (Test-OllamaApi)
    $modelAvailable = $false
    if ($ollamaRunning) {
        $models = & ollama list 2>$null
        $modelAvailable = $LASTEXITCODE -eq 0 -and ($models -match "(?m)^gemma3:latest\s")
    }

    Write-Host "Docker engine:        $(if ($dockerRunning) { 'running' } else { 'unavailable' })"
    Write-Host "Ollama API:           $(if ($ollamaRunning) { 'reachable' } else { 'unavailable' })"
    Write-Host "gemma3:latest:        $(if ($modelAvailable) { 'available' } else { 'unavailable' })"
    if ($dockerRunning) {
        Write-Host ""
        & docker compose --profile public ps
    }
    try {
        $health = Get-JsonEndpoint "$localUrl/healthz" 3
        Write-Host "Local health:         $($health.status)"
    } catch {
        Write-Host "Local health:         unavailable"
    }
    try {
        $ready = Get-JsonEndpoint "$localUrl/readyz" 3
        Write-Host "Local readiness:      $($ready.ready) ($($ready.indexed_chunks) chunks)"
    } catch {
        Write-Host "Local readiness:      unavailable"
    }

    $savedUrl = $null
    if (Test-Path -LiteralPath $publicUrlFile) {
        $candidate = (Get-Content -LiteralPath $publicUrlFile -Raw).Trim()
        if (Test-PublicUrlFormat $candidate) { $savedUrl = $candidate }
    }
    Write-Host "Validated public URL: $(if ($savedUrl) { $savedUrl } else { 'unavailable' })"
    if ($savedUrl) {
        try {
            $publicHealth = Get-JsonEndpoint "$savedUrl/healthz" 5
            Write-Host "Public health:        $($publicHealth.status)"
        } catch {
            Write-Host "Public health:        unavailable"
        }
    }
    if ($dockerRunning) {
        $connection = & docker compose logs --no-color --tail=80 cloudflared 2>$null |
            Select-String -Pattern "Registered tunnel connection|control stream encountered|Serve tunnel error" |
            Select-Object -Last 1
        Write-Host "Tunnel state:         $(if ($connection) { $connection.Line.Trim() } else { 'no connection log found' })"
    }
}

function Open-CurrentPublicDemo {
    if (-not (Test-Path -LiteralPath $publicUrlFile)) {
        throw "No validated public URL is saved. Run the public demo first."
    }
    $url = (Get-Content -LiteralPath $publicUrlFile -Raw).Trim()
    if (-not (Test-PublicUrlFormat $url)) {
        throw "The saved public URL is not a valid HTTPS trycloudflare.com address."
    }
    try {
        $health = Get-JsonEndpoint "$url/healthz" 8
        if ($health.status -ne "ok") { throw "Health response was not ok." }
    } catch {
        throw "The saved public URL is not currently reachable. Start the public demo to create a new URL."
    }
    Start-Process $url
    Write-Host "Opened $url"
}

Push-Location $repoRoot
try {
    switch ($Action) {
        "start"  { Start-Demo -EnablePublic ([bool]$Public) }
        "stop"   { Stop-Demo }
        "status" { Show-Status }
        "open"   { Open-CurrentPublicDemo }
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
