param(
    [string]$PackageName = "com.DefaultCompany.test2",
    [string]$LocalDir = "F:\TestData",
    [int]$PollSeconds = 2,
    [string]$Serial = "",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $msg"
}

function Invoke-Adb([string[]]$args) {
    if ([string]::IsNullOrWhiteSpace($Serial)) {
        return & adb @args 2>&1
    }

    return & adb -s $Serial @args 2>&1
}

function Get-RemoteDirs {
    return @(
        "/sdcard/Android/media/$PackageName/TestData",
        "/storage/emulated/0/Android/media/$PackageName/TestData",
        "/sdcard/Android/data/$PackageName/files/TestData",
        "/storage/emulated/0/Android/data/$PackageName/files/TestData",
        "/sdcard/Download/TestData"
    )
}

function Find-ReadableRemoteDir {
    foreach ($dir in Get-RemoteDirs) {
        try {
            $out = Invoke-Adb @("shell", "ls", "-d", $dir)
            if ($LASTEXITCODE -eq 0) {
                return $dir
            }
        }
        catch {
            # ignore candidate errors
        }
    }

    return $null
}

function Get-RemoteCsvFiles([string]$remoteDir) {
    $probe = "ls -1 $remoteDir/*.csv 2>/dev/null"
    $out = Invoke-Adb @("shell", "sh", "-c", $probe)
    if ($LASTEXITCODE -ne 0) {
        return @()
    }

    return $out |
        ForEach-Object { "$($_)".Trim() } |
        Where-Object { $_ -and $_.EndsWith(".csv") }
}

function Pull-NewFiles([string]$remoteDir, [System.Collections.Generic.HashSet[string]]$synced) {
    $files = Get-RemoteCsvFiles -remoteDir $remoteDir
    foreach ($remoteFile in $files) {
        if ($synced.Contains($remoteFile)) {
            continue
        }

        $name = [System.IO.Path]::GetFileName($remoteFile)
        $localPath = Join-Path $LocalDir $name

        if (Test-Path $localPath) {
            $synced.Add($remoteFile) | Out-Null
            continue
        }

        Write-Info "Pulling $name"
        $pullOut = Invoke-Adb @("pull", $remoteFile, $localPath)
        if ($LASTEXITCODE -eq 0) {
            $synced.Add($remoteFile) | Out-Null
            Write-Info "Saved -> $localPath"
        }
        else {
            Write-Info "Pull failed: $remoteFile"
            $pullOut | ForEach-Object { Write-Host $_ }
        }
    }
}

try {
    $null = & adb version 2>$null
}
catch {
    throw "adb is not found in PATH. Please install Android platform-tools and add adb to PATH."
}

if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
}

Write-Info "Local target: $LocalDir"
Write-Info "Package: $PackageName"
if (-not [string]::IsNullOrWhiteSpace($Serial)) {
    Write-Info "Device serial: $Serial"
}

$remoteDir = Find-ReadableRemoteDir
if (-not $remoteDir) {
    throw "Cannot locate remote TestData folder. Start app once, then retry. Expected under Android/data or Android/media path."
}

Write-Info "Remote source: $remoteDir"

$synced = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)

# Initial sync
Pull-NewFiles -remoteDir $remoteDir -synced $synced

if ($Once) {
    Write-Info "One-time sync complete."
    exit 0
}

Write-Info "Watching for new CSV files... Press Ctrl+C to stop."
while ($true) {
    Pull-NewFiles -remoteDir $remoteDir -synced $synced
    Start-Sleep -Seconds $PollSeconds
}
