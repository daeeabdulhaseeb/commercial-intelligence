# Rebuilds the public demo (synthetic data only: the full BI suite + predictive layer), runs the privacy scan and pushes to GitHub Pages.
# Stops before pushing if the scan finds any customer / SE / project name from the private data.
Set-Location $PSScriptRoot
. .\_py.ps1
$py = Get-Py
Test-Packages $py
$env:PYTHONUNBUFFERED = '1'
# refresh the protected-term list first: the build scrubs against it, the scan checks against it
if (Test-Path private\config.json) { & $py scan_public.py --refresh private\config.json }
Write-Host 'Rebuilding the public demo (full BI suite on the synthetic company)...' -ForegroundColor Cyan
& $py suite\build_suite.py
if ($LASTEXITCODE -ne 0) { Write-Host 'Demo build failed.' -ForegroundColor Red; exit 1 }
Write-Host 'Privacy scan...' -ForegroundColor Cyan
& $py scan_public.py
if ($LASTEXITCODE -ne 0) { Write-Host 'Privacy scan failed - nothing pushed.' -ForegroundColor Red; exit 1 }
if (-not (git config user.email)) {
    git config user.name 'Mohammed Abdul Haseeb'
    git config user.email 'daeeabdulhaseeb@users.noreply.github.com'
}
git rm -r -q --cached --ignore-unmatch suite | Out-Null
git add -A
git commit -m "Rebuild demo"
git push
if ($LASTEXITCODE -eq 0) { Write-Host 'Pushed. Live in ~1 minute: https://daeeabdulhaseeb.github.io/commercial-intelligence/' -ForegroundColor Green }
