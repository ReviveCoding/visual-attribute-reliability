#requires -Version 5.1
<#!
.SYNOPSIS
Runs read-only pytest evidence-contract tests for the frozen Fashionpedia release.

.DESCRIPTION
This driver does not import a model runtime, load Fashionpedia raw data, run inference,
train, re-score validation, or select a checkpoint. It verifies only the local evidence
freeze and staged source snapshots through JSON contracts and SHA-256 hashes.
#>

& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    $RepoRoot = "$HOME\Downloads\Amazon07_fashionpedia-attribute-reliability"
    $OutputRoot = "$HOME\ML_Outputs\Amazon07_fashionpedia-attribute-reliability"
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $ReleaseName = "fashionpedia-neckline-attribute-reliability-v1.0.0-local-evidence"
    $ReleaseRoot = Join-Path $RepoRoot "dist\$ReleaseName"
    $TestFile = Join-Path $RepoRoot "tests\test_release_evidence_contracts.py"

    foreach ($Path in @($RepoRoot, $VenvPython, $ReleaseRoot, $TestFile)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Required test-hardening path is missing: $Path"
        }
    }

    $Timestamp = Get-Date -Format "yyyyMMddTHHmmss"
    $LogRoot = Join-Path $OutputRoot "logs\fashionpedia_test_hardening_v1"
    $JUnitXml = Join-Path $LogRoot "$Timestamp`_release_evidence_contracts.junit.xml"
    $StdoutLog = Join-Path $LogRoot "$Timestamp`_release_evidence_contracts.stdout.log"
    $StderrLog = Join-Path $LogRoot "$Timestamp`_release_evidence_contracts.stderr.log"
    $SummaryJson = Join-Path $LogRoot "$Timestamp`_release_evidence_contracts.summary.json"

    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "FASHIONPEDIA RELEASE EVIDENCE CONTRACT TESTS"
    Write-Host "============================================================"
    Write-Host "Release root: $ReleaseRoot"
    Write-Host "Test file: $TestFile"
    Write-Host "No model loading, inference, training, raw-dataset access, or validation rescoring will run."

    $env:AMAZON07_REPO_ROOT = $RepoRoot
    $env:AMAZON07_RELEASE_ROOT = $ReleaseRoot

    & $VenvPython -m pytest -q $TestFile "--junitxml=$JUnitXml" 1> $StdoutLog 2> $StderrLog
    $ExitCode = $LASTEXITCODE

    Write-Host ""
    Write-Host "===== PYTEST STDOUT ====="
    if (Test-Path -LiteralPath $StdoutLog) {
        Get-Content -LiteralPath $StdoutLog
    }

    Write-Host ""
    Write-Host "===== PYTEST STDERR ====="
    if (Test-Path -LiteralPath $StderrLog) {
        Get-Content -LiteralPath $StderrLog
    }

    if ($ExitCode -ne 0) {
        throw "Release evidence-contract tests failed with exit code $ExitCode."
    }

    if (-not (Test-Path -LiteralPath $JUnitXml)) {
        throw "pytest completed without JUnit XML: $JUnitXml"
    }

    [System.IO.File]::WriteAllText(
        $SummaryJson,
        (@{
            status = "PASS"
            suite = "fashionpedia_release_evidence_contracts_v1"
            release_root = $ReleaseRoot
            test_file = $TestFile
            junit_xml = $JUnitXml
            stdout_log = $StdoutLog
            stderr_log = $StderrLog
            no_model_runtime_loaded = $true
            no_inference_performed = $true
            no_training_performed = $true
            no_validation_rescoring = $true
        } | ConvertTo-Json -Depth 10),
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RELEASE EVIDENCE CONTRACT TESTS QUALIFIED"
    Write-Host "============================================================"
    Write-Host "STATUS: PASS"
    Write-Host "JUNIT_XML: $JUnitXml"
    Write-Host "SUMMARY_JSON: $SummaryJson"
}
