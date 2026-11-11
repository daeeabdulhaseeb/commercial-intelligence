# Builds the private dashboard + predictive.json from company data and opens it. Output stays in private\ (git-ignored).
Set-Location $PSScriptRoot
. .\_py.ps1
$py = Get-Py
Test-Packages $py
Write-Host 'Building the private dashboard (real data, takes 1-3 minutes)...' -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = '1'
& $py -m ci.build --private private\config.json
if ($LASTEXITCODE -ne 0) { Write-Host 'Build failed - copy the error above to Claude.' -ForegroundColor Red; exit 1 }
Write-Host 'Done: private\site\index.html and private\site\predictive.json' -ForegroundColor Green
Start-Process (Join-Path $PSScriptRoot 'private\site\index.html')
