param(
    [string]$ProjectRoot = "E:\Reserch\AI\Natality_Comorbidity_SSL_20260525",
    [int[]]$Years = @(2016,2017,2018,2019,2020,2021,2022,2023,2024),
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"
)

$ErrorActionPreference = "Stop"
$script = Join-Path $ProjectRoot "scripts\01_prepare\build_analytic_cohort_2024.py"
$processed = Join-Path $ProjectRoot "data\processed"
New-Item -ItemType Directory -Force -Path $processed | Out-Null

foreach ($year in $Years) {
    $out = Join-Path $processed ("nat{0}_analytic_cohort.parquet" -f $year)
    if (Test-Path -LiteralPath $out) {
        $item = Get-Item -LiteralPath $out
        if ($item.Length -gt 0) {
            Write-Host "skip $year existing $($item.Length) bytes"
            continue
        }
    }
    Write-Host "build analytic cohort $year"
    & $Python $script --year $year --chunk-size 100000 --progress-every 1000000
}

Get-ChildItem -LiteralPath $processed -Filter "nat*_analytic_cohort.parquet" |
    Sort-Object Name |
    Select-Object Name,Length,LastWriteTime
