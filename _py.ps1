# Shared helper: finds a working Python (the Windows Store "python" alias does nothing) and checks packages.
function Get-Py {
    foreach ($c in @('py', 'python', 'python3')) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        if ($cmd.Source -like '*WindowsApps*') { continue }      # Microsoft Store stub - not a real Python
        $v = & $c -c "import sys; print(sys.version.split()[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { Write-Host "Python $v ($($cmd.Source))"; return $c }
    }
    Write-Host 'No working Python found. Install Python 3.11+ from python.org (tick "Add to PATH"), then reopen PowerShell.' -ForegroundColor Red
    exit 1
}
function Test-Packages($py) {
    & $py -c "import pandas, numpy, sklearn, scipy, pyarrow, openpyxl" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing missing packages (one time, ~2 min)...' -ForegroundColor Yellow
        & $py -m pip install -r requirements.txt openpyxl
        if ($LASTEXITCODE -ne 0) { Write-Host 'pip install failed - see the message above.' -ForegroundColor Red; exit 1 }
    }
}
