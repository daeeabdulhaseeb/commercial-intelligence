# Rebuild the public demo, run the privacy scan and push. Stops if the scan finds anything.
Set-Location $PSScriptRoot
python -m ci.build
if ($LASTEXITCODE -ne 0) { exit 1 }
python scan_public.py --refresh private\config.json
python scan_public.py
if ($LASTEXITCODE -ne 0) { Write-Host 'Privacy scan failed - nothing pushed.' -ForegroundColor Red; exit 1 }
git add -A
git commit -m "Rebuild demo"
git push
