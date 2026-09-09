param(
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputParent = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null

Push-Location $repoRoot
try {
    $sourceCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "git rev-parse failed"
    }

    & git archive --format=tar.gz "--output=$outputPath" $sourceCommit -- `
        src/molgap `
        experiments/qm9_architecture/qm9_fragment_state.py `
        experiments/qm9_architecture/fragment_state_protocol.md `
        platforms/scnet/qm9_fragment_state
    if ($LASTEXITCODE -ne 0) {
        throw "git archive failed"
    }

    $entries = @(& tar -tzf $outputPath)
    if ($LASTEXITCODE -ne 0) {
        throw "archive listing failed"
    }
    foreach ($required in @(
        "src/molgap/__init__.py",
        "src/molgap/constants.py",
        "src/molgap/qm9_data.py",
        "src/molgap/qm9_fragment_state.py",
        "experiments/qm9_architecture/qm9_fragment_state.py"
    )) {
        if ($required -notin $entries) {
            throw "runtime archive is missing $required"
        }
    }

    $hash = Get-FileHash $outputPath -Algorithm SHA256
    [pscustomobject]@{
        source_commit = $sourceCommit
        path = $outputPath
        bytes = (Get-Item $outputPath).Length
        sha256 = $hash.Hash.ToLowerInvariant()
    } | ConvertTo-Json
}
finally {
    Pop-Location
}
