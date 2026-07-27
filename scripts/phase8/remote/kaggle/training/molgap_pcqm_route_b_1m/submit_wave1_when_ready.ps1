param(
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../../../../..")).Path
$Kaggle = Join-Path $ProjectRoot ".venv/Scripts/kaggle.exe"
$Packages = Join-Path $PSScriptRoot "packages"
$StatePath = Join-Path $ProjectRoot "results/phase8/pcqm_route_b_1m/wave1_submission.json"
$LogPath = Join-Path $ProjectRoot "results/phase8/pcqm_route_b_1m/wave1_submit.log"
$SlotCooldownSeconds = 180
$Blockers = @(
    "nothingnessvoid/molgap-qm9-encoder-seeds",
    "nothingnessvoid/molgap-qm9-conformer-scaling"
)

$Jobs = @(
    @{
        Name = "gps9"
        Slug = "nothingnessvoid/molgap-pcqm-route-b-gps9-1m-r1-20260727"
        Package = Join-Path $Packages "gps9"
        MaximumRunningBlockers = 1
        Datasets = @(
            "nothingnessvoid/molgap-pcqm-route-b-gps-1m-20260727",
            "nothingnessvoid/molgap-pcqm-route-b-warmstarts-20260727"
        )
    },
    @{
        Name = "augmented_schnet"
        Slug = "nothingnessvoid/molgap-pcqm-route-b-augmented-schnet-1m-r1-20260727"
        Package = Join-Path $Packages "augmented_schnet"
        MaximumRunningBlockers = 0
        Datasets = @(
            "nothingnessvoid/molgap-pcqm-route-b-primary-1m-20260727",
            "nothingnessvoid/molgap-pcqm-route-b-secondary-1m-20260727",
            "nothingnessvoid/molgap-pcqm-route-b-warmstarts-20260727"
        )
    }
)

function Write-RunLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $LogPath -Value $line
}

function Test-Dataset([string]$Slug) {
    $null = & $Kaggle datasets files $Slug --csv 2>&1
    return $LASTEXITCODE -eq 0
}

function Test-Kernel([string]$Slug) {
    $null = & $Kaggle kernels status $Slug 2>&1
    return $LASTEXITCODE -eq 0
}

function Get-RunningBlockerCount {
    $Running = 0
    foreach ($Slug in $Blockers) {
        $Output = & $Kaggle kernels status $Slug 2>&1
        if ($LASTEXITCODE -ne 0 -or ($Output -join "`n") -match "RUNNING") {
            $Running += 1
        }
    }
    return $Running
}

function Save-State([hashtable]$Submitted) {
    $temporary = "$StatePath.tmp"
    @{
        format = "molgap-pcqm-route-b-wave1-submission-v1"
        updated_at = (Get-Date -Format o)
        submitted = $Submitted
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

$Submitted = @{}
$ReadySince = @{}
foreach ($Job in $Jobs) {
    $Submitted[$Job.Name] = Test-Kernel $Job.Slug
    $ReadySince[$Job.Name] = $null
}
Save-State $Submitted
Write-RunLog "wave1 submitter started"

while ($Submitted.Values -contains $false) {
    foreach ($Job in $Jobs) {
        if ($Submitted[$Job.Name]) {
            continue
        }
        $Ready = $true
        foreach ($Dataset in $Job.Datasets) {
            if (-not (Test-Dataset $Dataset)) {
                $Ready = $false
                Write-RunLog "$($Job.Name) waits for dataset $Dataset"
                break
            }
        }
        if (-not $Ready) {
            $ReadySince[$Job.Name] = $null
            continue
        }

        $RunningBlockers = Get-RunningBlockerCount
        if ($RunningBlockers -gt $Job.MaximumRunningBlockers) {
            $ReadySince[$Job.Name] = $null
            Write-RunLog "$($Job.Name) waits for a GPU slot; blockers=$RunningBlockers"
            continue
        }
        if ($null -eq $ReadySince[$Job.Name]) {
            $ReadySince[$Job.Name] = Get-Date
            Write-RunLog "$($Job.Name) GPU slot cooldown started"
            continue
        }
        $Cooldown = ((Get-Date) - $ReadySince[$Job.Name]).TotalSeconds
        if ($Cooldown -lt $SlotCooldownSeconds) {
            Write-RunLog "$($Job.Name) waits for GPU slot cooldown"
            continue
        }

        Write-RunLog "attempting $($Job.Name)"
        $Output = & $Kaggle kernels push -p $Job.Package 2>&1
        $Exists = Test-Kernel $Job.Slug
        if ($Exists) {
            $Submitted[$Job.Name] = $true
            Save-State $Submitted
            Write-RunLog "submitted $($Job.Slug)"
        } elseif (($Output -join "`n") -match "Maximum batch GPU session count") {
            Write-RunLog "$($Job.Name) waits for a GPU slot"
        } else {
            Write-RunLog "$($Job.Name) push failed: $($Output -join ' | ')"
        }
    }
    if ($Submitted.Values -contains $false) {
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-RunLog "wave1 submitter complete"
