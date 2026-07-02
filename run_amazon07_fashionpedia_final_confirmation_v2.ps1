param(
    [string]$DevelopmentRunRoot = "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\runs\siglip2_lora_matched_head_development_v2_bf16_20260701T195803"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$RepoRoot = "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability"
$OutputRoot = "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability"
$DatasetRoot = "C:\Users\bjw-0\Downloads\Project_Data\Fashionpedia"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunnerScript = Join-Path $RepoRoot "scripts\final_confirmation_fashionpedia_neckline_v2.py"
$TaskConfig = Join-Path $RepoRoot "configs\fashionpedia_neckline_attribute_v1.json"
$CacheRoot = "C:\Users\bjw-0\ML_Cache\huggingface"
$ManifestRoot = Join-Path $OutputRoot "manifests\fashionpedia_neckline_attribute_v1b_group_stratified"
$TrainManifest = Join-Path $ManifestRoot "train_manifest.csv"
$DevManifest = Join-Path $ManifestRoot "dev_manifest.csv"
$TrainAnnotation = Join-Path $DatasetRoot "instances_attributes_train2020.json"
$ValidationAnnotation = Join-Path $DatasetRoot "instances_attributes_val2020.json"
$ValidationImageRoot = Join-Path $DatasetRoot "val_test\test"
$LogRoot = Join-Path $OutputRoot "logs\fashionpedia_final_confirmation_v2"

$ExpectedTaskConfigSha256 = "d4ff99dbb11098249b9d26713d67227dcfe2899c1826db8d933deaab0787ad12"

foreach ($Path in @(
    $RepoRoot,
    $OutputRoot,
    $DatasetRoot,
    $VenvPython,
    $RunnerScript,
    $TaskConfig,
    $DevelopmentRunRoot,
    (Join-Path $DevelopmentRunRoot "run_summary.json"),
    (Join-Path $DevelopmentRunRoot "config.json"),
    $TrainManifest,
    $DevManifest,
    $TrainAnnotation,
    $ValidationAnnotation,
    $ValidationImageRoot
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path is missing: $Path"
    }
}

$ActualTaskConfigSha256 = (
    Get-FileHash -LiteralPath $TaskConfig -Algorithm SHA256
).Hash.ToLowerInvariant()

if ($ActualTaskConfigSha256 -ne $ExpectedTaskConfigSha256) {
    throw @"
Immutable source task-config hash mismatch.

Path:     $TaskConfig
Expected: $ExpectedTaskConfigSha256
Actual:   $ActualTaskConfigSha256
"@
}

Set-Location -LiteralPath $RepoRoot
[System.Environment]::CurrentDirectory = $RepoRoot
New-Item -ItemType Directory -Force -Path $LogRoot, $CacheRoot | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "FINAL CONFIRMATION V2 PRE-FLIGHT"
Write-Host "============================================================"
Write-Host "Development run: $DevelopmentRunRoot"
Write-Host "Immutable task config: $TaskConfig"
Write-Host "Task config SHA256: $ActualTaskConfigSha256"
Write-Host "Validation annotation: $ValidationAnnotation"
Write-Host "Validation images: $ValidationImageRoot"
Write-Host "Runner: $RunnerScript"

$CompileLog = Join-Path $LogRoot "final_confirmation_v2_py_compile.log"
& $VenvPython -m py_compile $RunnerScript 1> $CompileLog 2>&1
if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath $CompileLog
    throw "Python runner compile check failed."
}

$ImportLog = Join-Path $LogRoot "final_confirmation_v2_import_check.log"
& $VenvPython -c "import peft, scipy, sklearn, torch, transformers; print('IMPORT_CHECK_PASS'); print(torch.__version__); print(peft.__version__)" 1> $ImportLog 2>&1
if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath $ImportLog
    throw "Runtime import check failed."
}
Get-Content -LiteralPath $ImportLog

$Timestamp = Get-Date -Format "yyyyMMddTHHmmssfff"
$FinalRunRoot = Join-Path $OutputRoot "runs\fashionpedia_neckline_final_confirmation_v2_$Timestamp"
if (Test-Path -LiteralPath $FinalRunRoot) {
    throw "Refusing to overwrite existing final-confirmation directory: $FinalRunRoot"
}

$ExecutionRoot = Join-Path $LogRoot "execution_$Timestamp"
if (Test-Path -LiteralPath $ExecutionRoot) {
    throw "Refusing to overwrite execution-log directory: $ExecutionRoot"
}
New-Item -ItemType Directory -Force -Path $ExecutionRoot | Out-Null

$StdoutLog = Join-Path $ExecutionRoot "final_confirmation_v2.stdout.log"
$StderrLog = Join-Path $ExecutionRoot "final_confirmation_v2.stderr.log"
$BatchPath = Join-Path $ExecutionRoot "run_final_confirmation_v2.cmd"

$Batch = @"
@echo off
setlocal DisableDelayedExpansion
cd /d "$RepoRoot"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

"$VenvPython" -u "$RunnerScript" ^
  --run-root "$DevelopmentRunRoot" ^
  --source-train-manifest "$TrainManifest" ^
  --source-dev-manifest "$DevManifest" ^
  --source-train-annotation "$TrainAnnotation" ^
  --task-config "$TaskConfig" ^
  --validation-annotation "$ValidationAnnotation" ^
  --validation-image-root "$ValidationImageRoot" ^
  --cache-dir "$CacheRoot" ^
  --output-dir "$FinalRunRoot" ^
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
Write-Host "RUN UNTOUCHED OFFICIAL FASHIONPEDIA VALIDATION CONFIRMATION V2"
Write-Host "============================================================"
Write-Host "Output root: $FinalRunRoot"

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
Write-Host "============================================================"
Write-Host "FINAL CONFIRMATION V2 STDOUT"
Write-Host "============================================================"
if (Test-Path -LiteralPath $StdoutLog) {
    Get-Content -LiteralPath $StdoutLog
}

Write-Host ""
Write-Host "============================================================"
Write-Host "FINAL CONFIRMATION V2 STDERR"
Write-Host "============================================================"
if (Test-Path -LiteralPath $StderrLog) {
    Get-Content -LiteralPath $StderrLog
}

if ($ExitCode -ne 0) {
    throw "Final confirmation V2 runner failed with exit code $ExitCode."
}

$SummaryPath = Join-Path $FinalRunRoot "final_confirmation_summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath)) {
    throw "Final confirmation V2 exited successfully but did not create summary: $SummaryPath"
}

$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
if ($Summary.status -ne "PASS") {
    throw "Final confirmation V2 summary status is not PASS: $($Summary.status)"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "OFFICIAL FASHIONPEDIA CONFIRMATION V2 QUALIFIED"
Write-Host "============================================================"

[PSCustomObject]@{
    Status                         = $Summary.status
    TargetCategoryId               = $Summary.data.target_category_id
    TargetCategoryName             = $Summary.data.target_category_name
    ValidationSamples              = $Summary.data.official_validation_samples
    ValidationImages               = $Summary.data.official_validation_images
    SourceImageOverlapCount        = $Summary.manifest_summary.official_validation.source_train_dev_image_id_overlap_count
    FrozenSelectedDevelopmentEpoch = $Summary.frozen_head.selected_development_epoch
    LoRASelectedDevelopmentEpoch   = $Summary.lora.selected_development_epoch
    FrozenRawMacroF1               = $Summary.frozen_head.raw_metrics.macro_f1
    LoRARawMacroF1                 = $Summary.lora.raw_metrics.macro_f1
    LoRAMinusFrozenRawMacroF1      = $Summary.lora_minus_frozen_head.raw.macro_f1
    FrozenRawECE                   = $Summary.frozen_head.raw_metrics.top_label_ece
    LoRARawECE                     = $Summary.lora.raw_metrics.top_label_ece
    FrozenCalibratedECE            = $Summary.frozen_head.temperature_calibrated_metrics.top_label_ece
    LoRACalibratedECE              = $Summary.lora.temperature_calibrated_metrics.top_label_ece
    FrozenTemperature              = $Summary.frozen_head.temperature_scaling.temperature
    LoRATemperature                = $Summary.lora.temperature_scaling.temperature
    FinalSummary                   = $SummaryPath
    StdoutLog                      = $StdoutLog
    StderrLog                      = $StderrLog
} | Format-List
