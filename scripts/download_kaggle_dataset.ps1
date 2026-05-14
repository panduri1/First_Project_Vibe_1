param(
    [string]$DatasetRef = "shivamb/machine-predictive-maintenance-classification"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutputDir = Join-Path $ProjectRoot "data\kaggle\machine_predictive_maintenance_classification"
$DocsDir = Join-Path $ProjectRoot "docs\260514"
$LogPath = Join-Path $DocsDir "kaggle_download_log.md"
$ZipPath = Join-Path $OutputDir "dataset.zip"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $DocsDir | Out-Null

function Write-DownloadLog {
    param([string[]]$Lines)
    Set-Content -Path $LogPath -Value $Lines -Encoding UTF8
}

function Read-DotEnv {
    param([string]$Path)
    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim("'").Trim('"')
        $values[$key] = $value
    }
    return $values
}

function Resolve-KaggleCredential {
    $envValues = Read-DotEnv (Join-Path $ProjectRoot ".env")
    $username = $env:KAGGLE_USERNAME
    $key = $env:KAGGLE_KEY
    $token = $env:KAGGLE_API_TOKEN

    if (-not $username -and $envValues.ContainsKey("KAGGLE_USERNAME")) {
        $username = $envValues["KAGGLE_USERNAME"]
    }
    if (-not $key -and $envValues.ContainsKey("KAGGLE_KEY")) {
        $key = $envValues["KAGGLE_KEY"]
    }
    if (-not $token -and $envValues.ContainsKey("KAGGLE_API_TOKEN")) {
        $token = $envValues["KAGGLE_API_TOKEN"]
    }

    if ($username -and $key) {
        return @{ Username = $username; Key = $key; Note = "Found KAGGLE_USERNAME/KAGGLE_KEY credentials." }
    }

    if ($token) {
        if ($username) {
            return @{ Username = $username; Key = $token; Note = "Found KAGGLE_USERNAME with KAGGLE_API_TOKEN used as key." }
        }

        try {
            $json = $token | ConvertFrom-Json
            if ($json.username -and $json.key) {
                return @{ Username = $json.username; Key = $json.key; Note = "Found KAGGLE_API_TOKEN JSON credentials." }
            }
        }
        catch {
            # Non-JSON token formats are handled below.
        }

        if ($token.Contains(":")) {
            $parts = $token.Split(":", 2)
            if ($parts[0] -and $parts[1]) {
                return @{ Username = $parts[0]; Key = $parts[1]; Note = "Found KAGGLE_API_TOKEN username:key credentials." }
            }
        }

        return @{ Username = $null; Key = $null; Note = "KAGGLE_API_TOKEN exists but could not be parsed as Kaggle username/key." }
    }

    return @{ Username = $null; Key = $null; Note = "No Kaggle credentials found in .env or process environment variables." }
}

$lines = @(
    "# Kaggle Download Log",
    "",
    "- Dataset: ``$DatasetRef``",
    "- Output directory: ``$OutputDir``"
)

try {
    $credential = Resolve-KaggleCredential
    $lines += "- $($credential.Note)"

    if (-not $credential.Username -or -not $credential.Key) {
        $lines += "- Status: download skipped because credentials were not available."
        Write-DownloadLog $lines
        exit 2
    }

    $pair = "{0}:{1}" -f $credential.Username, $credential.Key
    $encoded = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
    $headers = @{ Authorization = "Basic $encoded" }
    $url = "https://www.kaggle.com/api/v1/datasets/download/$DatasetRef"

    Invoke-WebRequest -Uri $url -Headers $headers -OutFile $ZipPath
    Expand-Archive -Path $ZipPath -DestinationPath $OutputDir -Force
    Remove-Item -Path $ZipPath -Force

    $files = Get-ChildItem -Path $OutputDir -File -Recurse |
        ForEach-Object { $_.FullName.Substring($OutputDir.Length + 1).Replace("\", "/") } |
        Sort-Object

    $lines += "- Status: download completed."
    $lines += "- Downloaded file count: $($files.Count)"
    foreach ($file in $files | Select-Object -First 20) {
        $lines += "  - ``$file``"
    }
    if ($files.Count -gt 20) {
        $lines += "  - ... and $($files.Count - 20) more files"
    }
    Write-DownloadLog $lines
}
catch {
    $lines += "- Status: download failed - ``$($_.Exception.GetType().Name)``"
    $lines += "- Detail: ``$($_.Exception.Message)``"
    Write-DownloadLog $lines
    exit 1
}
