& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    $RepoRoot = "$HOME\Downloads\Amazon07_fashionpedia-attribute-reliability"
    $OutputRoot = "$HOME\ML_Outputs\Amazon07_fashionpedia-attribute-reliability"
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $Runner = Join-Path $RepoRoot "scripts\analyze_fashionpedia_post_validation_v1.py"
    $AnalysisLogRoot = Join-Path $OutputRoot "logs\fashionpedia_post_validation_analysis_v1"
    $RunsRoot = Join-Path $OutputRoot "runs"

    foreach ($Path in @($RepoRoot, $OutputRoot, $VenvPython, $Runner, $RunsRoot)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Required path is missing: $Path"
        }
    }

    $ConfirmationRun = @(
        Get-ChildItem -LiteralPath $RunsRoot -Directory -Filter "fashionpedia_neckline_final_confirmation_v2_*" |
        Sort-Object LastWriteTimeUtc -Descending
    ) | Select-Object -First 1

    if ($null -eq $ConfirmationRun) {
        throw "Could not locate a Fashionpedia final-confirmation V2 run under: $RunsRoot"
    }

    $ConfirmationSummary = Join-Path $ConfirmationRun.FullName "final_confirmation_summary.json"
    if (-not (Test-Path -LiteralPath $ConfirmationSummary)) {
        throw "Final confirmation summary is missing: $ConfirmationSummary"
    }

    $FinalSummary = Get-Content -LiteralPath $ConfirmationSummary -Raw | ConvertFrom-Json
    if ($FinalSummary.status -ne "PASS") {
        throw "Final confirmation status is not PASS: $($FinalSummary.status)"
    }

    New-Item -ItemType Directory -Force -Path $AnalysisLogRoot | Out-Null

    $Timestamp = Get-Date -Format "yyyyMMddTHHmmss"
    $AnalysisRoot = Join-Path $OutputRoot "analysis\fashionpedia_neckline_post_validation_analysis_v1_$Timestamp"
    $StdoutLog = Join-Path $AnalysisLogRoot "$Timestamp`_post_validation_analysis.stdout.log"
    $StderrLog = Join-Path $AnalysisLogRoot "$Timestamp`_post_validation_analysis.stderr.log"
    $BatchPath = Join-Path $AnalysisLogRoot "$Timestamp`_run_post_validation_analysis.cmd"

    if (Test-Path -LiteralPath $AnalysisRoot) {
        throw "Refusing to overwrite analysis output directory: $AnalysisRoot"
    }

    $DependencyCheck = @"
import matplotlib
import numpy
import PIL
print('DEPENDENCY_CHECK_PASS')
print('matplotlib=' + matplotlib.__version__)
print('numpy=' + numpy.__version__)
print('pillow=' + PIL.__version__)
"@

    $DependencyScript = Join-Path $AnalysisLogRoot "$Timestamp`_dependency_check.py"
    [System.IO.File]::WriteAllText(
        $DependencyScript,
        $DependencyCheck,
        [System.Text.UTF8Encoding]::new($false)
    )

    $DependencyStdout = Join-Path $AnalysisLogRoot "$Timestamp`_dependency_check.stdout.log"
    $DependencyStderr = Join-Path $AnalysisLogRoot "$Timestamp`_dependency_check.stderr.log"

    & $env:ComSpec /d /c "`"$VenvPython`" -u `"$DependencyScript`" 1>`"$DependencyStdout`" 2>`"$DependencyStderr`""
    $DependencyExitCode = $LASTEXITCODE

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "POST-VALIDATION ANALYSIS PRE-FLIGHT"
    Write-Host "============================================================"
    Write-Host "Confirmation run: $($ConfirmationRun.FullName)"
    Write-Host "Confirmation summary: $ConfirmationSummary"
    Write-Host "Runner: $Runner"
    Write-Host ""
    if (Test-Path -LiteralPath $DependencyStdout) { Get-Content -LiteralPath $DependencyStdout }
    if (Test-Path -LiteralPath $DependencyStderr) { Get-Content -LiteralPath $DependencyStderr }

    if ($DependencyExitCode -ne 0) {
        throw @"
Missing analysis dependencies in project venv. Install them with:

& `"$VenvPython`" -m pip install --isolated --no-input --disable-pip-version-check matplotlib

Then run this driver again.
"@
    }

    & $VenvPython -m py_compile $Runner
    if ($LASTEXITCODE -ne 0) {
        throw "Post-validation analysis Python compile check failed."
    }

    $Batch = @"
@echo off
setlocal DisableDelayedExpansion
cd /d "$RepoRoot"
set PYTHONUTF8=1
"$VenvPython" -u "$Runner" ^
  --confirmation-summary "$ConfirmationSummary" ^
  --output-dir "$AnalysisRoot" ^
  --max-examples-per-group 6 ^
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
    Write-Host "RUN FASHIONPEDIA POST-VALIDATION ANALYSIS"
    Write-Host "============================================================"
    Write-Host "No model loading, inference, training, validation re-scoring, or selection will run."
    Write-Host "Analysis root: $AnalysisRoot"

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
    Write-Host "===== POST-VALIDATION ANALYSIS STDOUT ====="
    if (Test-Path -LiteralPath $StdoutLog) { Get-Content -LiteralPath $StdoutLog }
    Write-Host ""
    Write-Host "===== POST-VALIDATION ANALYSIS STDERR ====="
    if (Test-Path -LiteralPath $StderrLog) { Get-Content -LiteralPath $StderrLog }

    if ($ExitCode -ne 0) {
        throw "Post-validation analysis runner failed with exit code $ExitCode."
    }

    $AnalysisManifest = Join-Path $AnalysisRoot "analysis_artifact_manifest.json"
    $AnalysisReport = Join-Path $AnalysisRoot "POST_VALIDATION_ANALYSIS_REPORT.md"
    if (-not (Test-Path -LiteralPath $AnalysisManifest)) {
        throw "Analysis completed without manifest: $AnalysisManifest"
    }
    if (-not (Test-Path -LiteralPath $AnalysisReport)) {
        throw "Analysis completed without report: $AnalysisReport"
    }

    $Result = Get-Content -LiteralPath $AnalysisManifest -Raw | ConvertFrom-Json
    if ($Result.status -ne "PASS") {
        throw "Analysis manifest status is not PASS: $($Result.status)"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "POST-VALIDATION ANALYSIS QUALIFIED"
    Write-Host "============================================================"
    [PSCustomObject]@{
        Status                       = $Result.status
        OfficialValidationSamples    = $Result.official_validation_samples
        NoModelRuntimeLoaded         = $Result.no_model_runtime_loaded
        NoInferencePerformed         = $Result.no_inference_performed
        NoTrainingPerformed          = $Result.no_training_performed
        NoSelectionPerformed         = $Result.no_checkpoint_or_hyperparameter_selection_performed
        AnalysisRoot                 = $AnalysisRoot
        AnalysisReport               = $AnalysisReport
        AnalysisManifest             = $AnalysisManifest
        StdoutLog                    = $StdoutLog
        StderrLog                    = $StderrLog
    } | Format-List
}
