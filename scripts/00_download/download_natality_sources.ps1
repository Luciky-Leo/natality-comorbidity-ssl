param(
    [string]$ProjectRoot = "E:\Reserch\AI\Natality_Comorbidity_SSL_20260525",
    [int[]]$Years = @(2016,2017,2018,2019,2020,2021,2022,2023,2024)
)

$ErrorActionPreference = "Stop"
$manifest = Join-Path $ProjectRoot "config\natality_download_manifest.csv"
$rawDir = Join-Path $ProjectRoot "data\raw"
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$rows = Import-Csv -LiteralPath $manifest | Where-Object { $Years -contains [int]$_.year }

foreach ($row in $rows) {
    foreach ($kind in @("guide", "us_data")) {
        if ($kind -eq "guide") {
            $url = $row.guide_url
            $file = $row.guide_file
        } else {
            $url = $row.us_data_url
            $file = $row.us_data_file
        }
        $target = Join-Path $rawDir $file
        if (Test-Path -LiteralPath $target) {
            $item = Get-Item -LiteralPath $target
            if ($item.Length -gt 0) {
                Write-Host "skip $($row.year) $kind $file $($item.Length) bytes"
                continue
            }
        }

        $temp = "$target.part"
        if (Test-Path -LiteralPath $temp) {
            Remove-Item -LiteralPath $temp -Force
        }
        Write-Host "download $($row.year) $kind $file"
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $temp
        Move-Item -LiteralPath $temp -Destination $target -Force
        $item = Get-Item -LiteralPath $target
        Write-Host "done $file $($item.Length) bytes"
    }
}

Get-ChildItem -LiteralPath $rawDir | Sort-Object Name | Select-Object Name,Length,LastWriteTime
