param(
    [string]$OutputDir = "data/raw_linked_birth_infant_death",
    [switch]$DataOnly,
    [switch]$DocsOnly
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$target = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $target | Out-Null

$files = @(
    @{
        Name = "24PE23CO_linkedUG.pdf"
        Url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/period-cohort-linked/24PE23CO_linkedUG.pdf"
        Kind = "documentation"
    },
    @{
        Name = "2024PE2023CO-PS.zip"
        Url = "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/dvs/period-cohort-linked/2024PE2023CO-PS.zip"
        Kind = "prescreen"
    },
    @{
        Name = "2024PE2023CO.zip"
        Url = "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/dvs/period-cohort-linked/2024PE2023CO.zip"
        Kind = "data"
    }
)

foreach ($file in $files) {
    if ($DocsOnly -and $file.Kind -eq "data") { continue }
    if ($DataOnly -and $file.Kind -eq "documentation") { continue }
    $destination = Join-Path $target $file.Name
    if (Test-Path $destination) {
        Write-Host "exists $destination"
        continue
    }
    Write-Host "download $($file.Url)"
    Invoke-WebRequest -Uri $file.Url -OutFile $destination
}

Write-Host "done $target"
