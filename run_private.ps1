# Builds the private dashboard from company data and opens it. Output stays in private\ (git-ignored).
Set-Location $PSScriptRoot
python -m ci.build --private private\config.json
if ($LASTEXITCODE -eq 0) { Start-Process (Join-Path $PSScriptRoot 'private\site\index.html') }
