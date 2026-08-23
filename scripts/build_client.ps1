<#
.SYNOPSIS
    Builds the Windows desktop client into a distributable zip.

.DESCRIPTION
    Produces dist\InterviewCopilot-windows-<version>.zip containing:

        InterviewOverlay.exe      the answer overlay
        InterviewCapture.exe      system audio capture
        interviewcopilot.json     config template the user fills in
        start.bat                 launches both
        README.txt                setup in five lines

    The config file ships EMPTY on purpose. Baking a server URL and token
    into the executables would mean rebuilding for every deployment and
    every token rotation; the client reads them at startup instead, so one
    build serves every user (see client\config.py).

.PARAMETER Version
    Release version. Defaults to the contents of the VERSION file.

.PARAMETER ApiUrl
    Optional. Pre-fills api_url in the shipped config, for when you are
    building for one known backend and want a zero-configuration download.

.EXAMPLE
    powershell -File scripts\build_client.ps1

.EXAMPLE
    powershell -File scripts\build_client.ps1 -ApiUrl "https://api.example.com"
#>

[CmdletBinding()]
param(
    [string]$Version,
    [string]$ApiUrl = ""
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $Version) {
    $versionFile = Join-Path $repo "VERSION"
    if (Test-Path $versionFile) {
        $Version = (Get-Content $versionFile -Raw).Trim()
    } else {
        $Version = "0.0.0"
    }
}

Write-Host "Building InterviewCopilot client $Version" -ForegroundColor Cyan

# --- preflight ---------------------------------------------------------

$pyinstaller = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pyinstaller) {
    throw "pyinstaller not found. Run: pip install pyinstaller"
}

foreach ($module in @("PySide6", "soundcard", "sounddevice", "websockets", "requests")) {
    & python -c "import $module" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing client dependency '$module'. Run: pip install -r requirements-client.txt"
    }
}

$staging = Join-Path $repo "build\client-stage"
$distDir = Join-Path $repo "dist"

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging -Force | Out-Null
New-Item -ItemType Directory -Path $distDir -Force | Out-Null

# --- executables -------------------------------------------------------

# --noconsole on the overlay only: it is a GUI and a console window behind
# it would be visible on screen share, which defeats the point. The capture
# side keeps its console because the audio level meter is how you diagnose
# "it isn't hearing anything".

Write-Host "  building InterviewOverlay.exe ..." -ForegroundColor DarkGray
& pyinstaller --noconfirm --clean --onefile --noconsole `
    --name InterviewOverlay `
    --paths client `
    --distpath $staging `
    --workpath (Join-Path $repo "build\pyi-overlay") `
    --specpath (Join-Path $repo "build") `
    (Join-Path $repo "client\overlay_app.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed on overlay_app.py" }

Write-Host "  building InterviewCapture.exe ..." -ForegroundColor DarkGray
& pyinstaller --noconfirm --clean --onefile `
    --name InterviewCapture `
    --paths client `
    --distpath $staging `
    --workpath (Join-Path $repo "build\pyi-capture") `
    --specpath (Join-Path $repo "build") `
    (Join-Path $repo "client\test_loopback_stream.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed on test_loopback_stream.py" }

# --- shipped files -----------------------------------------------------

$config = [ordered]@{
    api_url    = $ApiUrl
    token      = ""
    session_id = ""
}
$config | ConvertTo-Json | Set-Content (Join-Path $staging "interviewcopilot.json") -Encoding utf8

@'
@echo off
rem Launches audio capture and the overlay together.
rem Settings come from interviewcopilot.json in this folder.
start "InterviewCopilot capture" InterviewCapture.exe
start "InterviewCopilot overlay" InterviewOverlay.exe
'@ | Set-Content (Join-Path $staging "start.bat") -Encoding ascii

@"
InterviewCopilot $Version

1. Open interviewcopilot.json in Notepad.
2. Paste in the api_url and token you were given, and pick a session_id
   (anything unique to you -- it keeps your resume and history separate).
3. Save, then run start.bat.
4. Drop your resume PDF onto the overlay window.
5. Play any audio and check the level meter in the capture window moves.

Connection not working? Run this in a terminal to see what settings the
client actually resolved, and which files it looked in:

    InterviewOverlay.exe --where-config

Capturing the wrong speakers? List and pick the right output device:

    InterviewCapture.exe --list-devices
    InterviewCapture.exe --device 3
"@ | Set-Content (Join-Path $staging "README.txt") -Encoding utf8

# --- package -----------------------------------------------------------

$zipName = "InterviewCopilot-windows-$Version.zip"
$zipPath = Join-Path $distDir $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath

$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Built $zipName ($sizeMb MB)" -ForegroundColor Green
Write-Host "  $zipPath"
Write-Host ""
Write-Host "Upload it to your release host, then point web/download.html at it." -ForegroundColor DarkGray
