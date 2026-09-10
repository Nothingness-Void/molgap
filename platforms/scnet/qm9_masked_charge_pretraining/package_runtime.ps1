param(
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) |
    Out-Null

Push-Location $repoRoot
try {
    $sourceCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed" }
    & git archive --format=tar.gz "--output=$outputPath" $sourceCommit -- `
        src/molgap `
        experiments/qm9_architecture/qm9_masked_charge_pretraining.py `
        experiments/qm9_architecture/masked_charge_pretraining_protocol.md `
        platforms/scnet/qm9_masked_charge_pretraining
    if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
    $entries = @(& tar -tzf $outputPath)
    if ($LASTEXITCODE -ne 0) { throw "archive listing failed" }
    foreach ($required in @(
        "src/molgap/constants.py",
        "src/molgap/qm9_charge_adapter.py",
        "src/molgap/qm9_masked_charge_pretraining.py",
        "src/molgap/screen_policy.py",
        "experiments/qm9_architecture/qm9_masked_charge_pretraining.py"
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
