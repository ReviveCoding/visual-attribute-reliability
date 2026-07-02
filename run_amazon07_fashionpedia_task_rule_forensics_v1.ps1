& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    $RepoRoot = `
        "$HOME\Downloads\Amazon07_fashionpedia-attribute-reliability"

    $OutputRoot = `
        "$HOME\ML_Outputs\Amazon07_fashionpedia-attribute-reliability"

    $VenvPython = Join-Path `
        $RepoRoot `
        ".venv\Scripts\python.exe"

    $ManifestRoot = Join-Path `
        $OutputRoot `
        "manifests\fashionpedia_neckline_attribute_v1b_group_stratified"

    $TrainManifest = Join-Path $ManifestRoot "train_manifest.csv"
    $DevManifest = Join-Path $ManifestRoot "dev_manifest.csv"

    $DatasetRoot = `
        "$HOME\Downloads\Project_Data\Fashionpedia"

    $TrainAnnotation = Join-Path `
        $DatasetRoot `
        "instances_attributes_train2020.json"

    $TaskConfig = Join-Path `
        $RepoRoot `
        "configs\fashionpedia_neckline_attribute_v1.json"

    $TaskAudit = Join-Path `
        $OutputRoot `
        "fashionpedia_neckline_train_only_audit_v1.json"

    $ScriptRoot = Join-Path $RepoRoot "scripts"
    $AuditScript = Join-Path `
        $ScriptRoot `
        "fashionpedia_neckline_source_rule_forensics_v1.py"

    $ForensicsRoot = Join-Path $OutputRoot "forensics"
    $LogRoot = Join-Path $OutputRoot "logs\fashionpedia_task_rule_forensics"

    foreach ($Path in @(
        $RepoRoot,
        $OutputRoot,
        $VenvPython,
        $TrainManifest,
        $DevManifest,
        $TrainAnnotation,
        $TaskConfig,
        $TaskAudit,
        $AuditScript
    )) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Required path is missing: $Path"
        }
    }

    Set-Location -LiteralPath $RepoRoot
    [System.Environment]::CurrentDirectory = $RepoRoot

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $ForensicsRoot, $LogRoot |
        Out-Null

    $Timestamp = Get-Date -Format "yyyyMMddTHHmmss"

    $OutputJson = Join-Path `
        $ForensicsRoot `
        "fashionpedia_neckline_source_rule_forensics_v1_$Timestamp.json"

    $BatchPath = Join-Path `
        $ForensicsRoot `
        "run_fashionpedia_neckline_source_rule_forensics_v1_$Timestamp.cmd"

    $StdoutLog = Join-Path `
        $LogRoot `
        "$Timestamp`_forensics.stdout.log"

    $StderrLog = Join-Path `
        $LogRoot `
        "$Timestamp`_forensics.stderr.log"

    $CompileStdout = Join-Path `
        $LogRoot `
        "$Timestamp`_compile.stdout.log"

    $CompileStderr = Join-Path `
        $LogRoot `
        "$Timestamp`_compile.stderr.log"

    & $VenvPython `
        -m `
        py_compile `
        $AuditScript `
        1> $CompileStdout `
        2> $CompileStderr

    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $CompileStderr) {
            Get-Content -LiteralPath $CompileStderr
        }
        throw "Forensics Python compile check failed."
    }

    $Batch = @"
@echo off
setlocal DisableDelayedExpansion
cd /d "$RepoRoot"
set PYTHONUTF8=1

"$VenvPython" -u "$AuditScript" ^
  --source-train-manifest "$TrainManifest" ^
  --source-dev-manifest "$DevManifest" ^
  --source-train-annotation "$TrainAnnotation" ^
  --task-config "$TaskConfig" ^
  --task-audit-json "$TaskAudit" ^
  --output-json "$OutputJson" ^
  1>"$StdoutLog" ^
  2>"$StderrLog"

exit /b %ERRORLEVEL%
"@

    [System.IO.File]::WriteAllText(
        $BatchPath,
        $Batch,
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RUN FASHIONPEDIA SOURCE-TASK RULE FORENSICS"
    Write-Host "============================================================"
    Write-Host "Scope: source train manifests plus source train annotation only"
    Write-Host "Official validation annotation: NOT OPENED"
    Write-Host "Official validation images: NOT SCANNED"
    Write-Host "Model: NOT LOADED"

    $PriorErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"
        & $env:ComSpec /d /c "call `"$BatchPath`""
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PriorErrorActionPreference
    }

    Write-Host ""
    Write-Host "===== FORENSICS STDOUT ====="
    if (Test-Path -LiteralPath $StdoutLog) {
        Get-Content -LiteralPath $StdoutLog
    }

    Write-Host ""
    Write-Host "===== FORENSICS STDERR ====="
    if (Test-Path -LiteralPath $StderrLog) {
        Get-Content -LiteralPath $StderrLog
    }

    if ($ExitCode -ne 0) {
        throw "Fashionpedia source-task forensics failed with exit code $ExitCode."
    }

    if (-not (Test-Path -LiteralPath $OutputJson)) {
        throw "Forensics exited with code 0 but did not create JSON: $OutputJson"
    }

    $Result = Get-Content -LiteralPath $OutputJson -Raw | ConvertFrom-Json

    if ($Result.status -ne "PASS") {
        throw "Forensics result status is not PASS: $($Result.status)"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "SOURCE-TASK RULE FORENSICS QUALIFIED"
    Write-Host "============================================================"
    Write-Host "FORENSICS_JSON: $OutputJson"
    Write-Host "NO OFFICIAL VALIDATION ANNOTATION WAS LOADED."
    Write-Host "NO OFFICIAL VALIDATION IMAGE WAS SCANNED."
    Write-Host "NO MODEL WAS LOADED."
}
