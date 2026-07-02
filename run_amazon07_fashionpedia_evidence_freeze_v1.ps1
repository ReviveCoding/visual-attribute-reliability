#requires -Version 5.1
<#
.SYNOPSIS
Freezes validated Fashionpedia neckline evidence into a local, hash-addressed release bundle.

.DESCRIPTION
This driver copies only code, manifests, experiment outputs, selected checkpoints, and dependency locks.
It does not load any model, run inference, access Fashionpedia raw images, or access raw annotation payloads.
It validates the known official-validation result before copying evidence.

Release status produced by this script:
LOCAL_EVIDENCE_FROZEN
This is not a hosted-CI or GitHub Release qualification.
#>

& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    $RepoRoot = "$HOME\Downloads\Amazon07_fashionpedia-attribute-reliability"
    $OutputRoot = "$HOME\ML_Outputs\Amazon07_fashionpedia-attribute-reliability"
    $ReleaseName = "fashionpedia-neckline-attribute-reliability-v1.0.0-local-evidence"

    $DevelopmentRun = Join-Path $OutputRoot `
        "runs\siglip2_lora_matched_head_development_v2_bf16_20260701T195803"

    $FinalConfirmationRun = Join-Path $OutputRoot `
        "runs\fashionpedia_neckline_final_confirmation_v2_20260701T215439857"

    $FrozenEmbeddingRun = Join-Path $OutputRoot `
        "runs\siglip2_frozen_neckline_v1_20260701T190347"

    $HogRun = Join-Path $OutputRoot `
        "runs\hog_neckline_v1_20260701T181320"

    $ManifestRoot = Join-Path $OutputRoot `
        "manifests\fashionpedia_neckline_attribute_v1b_group_stratified"

    $ForensicsJson = Join-Path $OutputRoot `
        "forensics\fashionpedia_neckline_source_rule_forensics_v1_20260701T214132.json"

    $TaskAuditJson = Join-Path $OutputRoot `
        "fashionpedia_neckline_train_only_audit_v1.json"

    $DistRoot = Join-Path $RepoRoot "dist"
    $ReleaseDirectory = Join-Path $DistRoot $ReleaseName
    $ReleaseZip = Join-Path $DistRoot "$ReleaseName.zip"
    $ReleaseZipSha = "$ReleaseZip.sha256"
    $ReleaseManifestSidecar = Join-Path $DistRoot "$ReleaseName.release_evidence_manifest.json"
    $ReleaseNotesSidecar = Join-Path $DistRoot "$ReleaseName.RELEASE_NOTES.md"

    function Assert-PathExists {
        param(
            [Parameter(Mandatory = $true)]
            [string]$Path,

            [Parameter(Mandatory = $true)]
            [string]$Label
        )

        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Required $Label is missing: $Path"
        }
    }

    function Assert-ApproxEqual {
        param(
            [Parameter(Mandatory = $true)]
            [double]$Actual,

            [Parameter(Mandatory = $true)]
            [double]$Expected,

            [Parameter(Mandatory = $true)]
            [string]$Label,

            [double]$Tolerance = 0.000000000001
        )

        if ([math]::Abs($Actual - $Expected) -gt $Tolerance) {
            throw "$Label mismatch. Expected=$Expected Actual=$Actual Tolerance=$Tolerance"
        }
    }

    function Copy-FileToStage {
        param(
            [Parameter(Mandatory = $true)]
            [string]$Source,

            [Parameter(Mandatory = $true)]
            [string]$RelativeDestination
        )

        Assert-PathExists -Path $Source -Label "source file"

        $Destination = Join-Path $ReleaseDirectory $RelativeDestination
        $Parent = Split-Path -Parent $Destination

        New-Item -ItemType Directory -Force -Path $Parent | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }

    function Copy-DirectoryToStage {
        param(
            [Parameter(Mandatory = $true)]
            [string]$Source,

            [Parameter(Mandatory = $true)]
            [string]$RelativeDestination
        )

        Assert-PathExists -Path $Source -Label "source directory"

        $SourceItem = Get-Item -LiteralPath $Source
        if (-not $SourceItem.PSIsContainer) {
            throw "Expected a directory but received a file: $Source"
        }

        $DestinationRoot = Join-Path $ReleaseDirectory $RelativeDestination
        New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

        $SourceRootWithSlash = $SourceItem.FullName.TrimEnd([char[]]'\') + '\'

        $Items = @(
            Get-ChildItem -LiteralPath $SourceItem.FullName -Recurse -Force |
            Sort-Object FullName
        )

        foreach ($Item in $Items) {
            $Relative = $Item.FullName.Substring($SourceRootWithSlash.Length)
            $Destination = Join-Path $DestinationRoot $Relative

            if ($Item.PSIsContainer) {
                New-Item -ItemType Directory -Force -Path $Destination | Out-Null
            }
            else {
                $Parent = Split-Path -Parent $Destination
                New-Item -ItemType Directory -Force -Path $Parent | Out-Null
                Copy-Item -LiteralPath $Item.FullName -Destination $Destination -Force
            }
        }
    }

    function Get-RelativeFileManifest {
        param(
            [Parameter(Mandatory = $true)]
            [string]$Root
        )

        $RootItem = Get-Item -LiteralPath $Root
        $RootWithSlash = $RootItem.FullName.TrimEnd([char[]]'\') + '\'

        $Rows = foreach ($File in @(
            Get-ChildItem -LiteralPath $RootItem.FullName -Recurse -File -Force |
            Sort-Object FullName
        )) {
            [PSCustomObject]@{
                relative_path = $File.FullName.Substring($RootWithSlash.Length).Replace('\\', '/')
                bytes = [int64]$File.Length
                sha256 = (
                    Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256
                ).Hash.ToLowerInvariant()
            }
        }

        return @($Rows)
    }

    function Get-OptionalGitState {
        param(
            [Parameter(Mandatory = $true)]
            [string]$RepositoryRoot
        )

        $GitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
        if ($null -eq $GitCommand) {
            return [PSCustomObject]@{
                available = $false
                commit = $null
                branch = $null
                dirty = $null
            }
        }

        try {
            $Commit = (& git.exe -C $RepositoryRoot rev-parse HEAD 2>$null).Trim()
            $Branch = (& git.exe -C $RepositoryRoot branch --show-current 2>$null).Trim()
            $StatusLines = @(& git.exe -C $RepositoryRoot status --porcelain 2>$null)

            return [PSCustomObject]@{
                available = $true
                commit = $(if ([string]::IsNullOrWhiteSpace($Commit)) { $null } else { $Commit })
                branch = $(if ([string]::IsNullOrWhiteSpace($Branch)) { $null } else { $Branch })
                dirty = ($StatusLines.Count -gt 0)
            }
        }
        catch {
            return [PSCustomObject]@{
                available = $true
                commit = $null
                branch = $null
                dirty = $null
            }
        }
    }

    foreach ($Path in @(
        $RepoRoot,
        $OutputRoot,
        $DevelopmentRun,
        $FinalConfirmationRun,
        $FrozenEmbeddingRun,
        $HogRun,
        $ManifestRoot,
        $ForensicsJson,
        $TaskAuditJson
    )) {
        Assert-PathExists -Path $Path -Label "required evidence artifact"
    }

    if (Test-Path -LiteralPath $ReleaseDirectory) {
        throw "Refusing to overwrite existing release directory: $ReleaseDirectory"
    }

    foreach ($Path in @($ReleaseZip, $ReleaseZipSha, $ReleaseManifestSidecar, $ReleaseNotesSidecar)) {
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing release asset: $Path"
        }
    }

    $DevelopmentSummaryPath = Join-Path $DevelopmentRun "run_summary.json"
    $FinalSummaryPath = Join-Path $FinalConfirmationRun "final_confirmation_summary.json"
    $FinalManifestSummaryPath = Join-Path $FinalConfirmationRun "manifest\official_validation_manifest_summary.json"
    $TaskConfigPath = Join-Path $RepoRoot "configs\fashionpedia_neckline_attribute_v1.json"
    $LoRAAdapterPath = Join-Path $DevelopmentRun "lora\best_lora_adapter"
    $LoRAHeadPath = Join-Path $DevelopmentRun "lora\best_head_state.pt"
    $FrozenHeadPath = Join-Path $DevelopmentRun "frozen_head\best_head_state.pt"

    foreach ($Path in @(
        $DevelopmentSummaryPath,
        $FinalSummaryPath,
        $FinalManifestSummaryPath,
        $TaskConfigPath,
        $LoRAAdapterPath,
        $LoRAHeadPath,
        $FrozenHeadPath
    )) {
        Assert-PathExists -Path $Path -Label "qualification artifact"
    }

    $AdapterConfigPath = Join-Path $LoRAAdapterPath "adapter_config.json"
    Assert-PathExists -Path $AdapterConfigPath -Label "LoRA adapter configuration"

    $AdapterWeights = @(
        Get-ChildItem -LiteralPath $LoRAAdapterPath -File -ErrorAction Stop |
        Where-Object {
            $_.Name -eq "adapter_model.safetensors" -or $_.Name -eq "adapter_model.bin"
        }
    )

    if ($AdapterWeights.Count -ne 1) {
        throw "Expected exactly one LoRA adapter weight file under: $LoRAAdapterPath"
    }

    $DevelopmentSummary = Get-Content -LiteralPath $DevelopmentSummaryPath -Raw | ConvertFrom-Json
    $FinalSummary = Get-Content -LiteralPath $FinalSummaryPath -Raw | ConvertFrom-Json
    $FinalManifestSummary = Get-Content -LiteralPath $FinalManifestSummaryPath -Raw | ConvertFrom-Json

    if ($DevelopmentSummary.status -ne "PASS") {
        throw "Development summary is not PASS: $($DevelopmentSummary.status)"
    }

    if ($FinalSummary.status -ne "PASS") {
        throw "Final confirmation summary is not PASS: $($FinalSummary.status)"
    }

    if ($FinalSummary.evaluation_scope -ne "untouched_official_fashionpedia_validation") {
        throw "Unexpected final evaluation scope: $($FinalSummary.evaluation_scope)"
    }

    if ($FinalSummary.selection_scope -ne "fixed_from_train_only_development_before_validation_access") {
        throw "Unexpected final selection scope: $($FinalSummary.selection_scope)"
    }

    if ($FinalSummary.data.target_category_id -ne 33 -or $FinalSummary.data.target_category_name -ne "neckline") {
        throw "Final confirmation target scope is not category 33 neckline."
    }

    if ($FinalSummary.data.official_validation_samples -ne 654 -or $FinalSummary.data.official_validation_images -ne 644) {
        throw "Unexpected final validation cohort size. Samples=$($FinalSummary.data.official_validation_samples) Images=$($FinalSummary.data.official_validation_images)"
    }

    $FinalCheckValues = @($FinalSummary.qualification_checks.PSObject.Properties | ForEach-Object { [bool]$_.Value })
    if ($FinalCheckValues -contains $false) {
        throw "One or more final confirmation qualification checks are false."
    }

    if ($FinalManifestSummary.status -ne "PASS") {
        throw "Final validation manifest summary is not PASS."
    }

    if ($FinalManifestSummary.source_train_rule_reproduction.pass -ne $true) {
        throw "Source task rule reconstruction did not pass."
    }

    if ($FinalManifestSummary.source_train_rule_reproduction.source_train_dev_pairs -ne 20800) {
        throw "Source task pair count is not 20800."
    }

    if ($FinalManifestSummary.official_validation.source_train_dev_image_id_overlap_count -ne 0) {
        throw "Official validation has source image overlap."
    }

    if ($FinalSummary.frozen_head.selected_development_epoch -ne 6) {
        throw "Unexpected frozen selected development epoch: $($FinalSummary.frozen_head.selected_development_epoch)"
    }

    if ($FinalSummary.lora.selected_development_epoch -ne 5) {
        throw "Unexpected LoRA selected development epoch: $($FinalSummary.lora.selected_development_epoch)"
    }

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.frozen_head.raw_metrics.macro_f1) `
        -Expected 0.5797703254640079 `
        -Label "Frozen official-validation Macro-F1"

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.lora.raw_metrics.macro_f1) `
        -Expected 0.6681089306597886 `
        -Label "LoRA official-validation Macro-F1"

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.lora_minus_frozen_head.raw.macro_f1) `
        -Expected 0.08833860519578074 `
        -Label "LoRA minus frozen official-validation Macro-F1"

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.frozen_head.raw_metrics.top_label_ece) `
        -Expected 0.08395556030105743 `
        -Label "Frozen raw ECE"

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.lora.raw_metrics.top_label_ece) `
        -Expected 0.06523039754742146 `
        -Label "LoRA raw ECE"

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.frozen_head.temperature_calibrated_metrics.top_label_ece) `
        -Expected 0.0297848879379599 `
        -Label "Frozen calibrated ECE"

    Assert-ApproxEqual `
        -Actual ([double]$FinalSummary.lora.temperature_calibrated_metrics.top_label_ece) `
        -Expected 0.03476740991759372 `
        -Label "LoRA calibrated ECE"

    New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $ReleaseDirectory | Out-Null

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "FASHIONPEDIA POST-VALIDATION EVIDENCE FREEZE"
    Write-Host "============================================================"
    Write-Host "Release directory: $ReleaseDirectory"
    Write-Host "No model inference, training, validation scoring, or dataset copying will run."

    Copy-FileToStage -Source $ForensicsJson -RelativeDestination "evidence\forensics\fashionpedia_neckline_source_rule_forensics_v1.json"
    Copy-FileToStage -Source $TaskAuditJson -RelativeDestination "evidence\task_audit\fashionpedia_neckline_train_only_audit_v1.json"
    Copy-DirectoryToStage -Source $ManifestRoot -RelativeDestination "evidence\manifests\fashionpedia_neckline_attribute_v1b_group_stratified"
    Copy-DirectoryToStage -Source $HogRun -RelativeDestination "evidence\runs\hog_neckline_v1"
    Copy-DirectoryToStage -Source $FrozenEmbeddingRun -RelativeDestination "evidence\runs\siglip2_frozen_neckline_v1"
    Copy-DirectoryToStage -Source $DevelopmentRun -RelativeDestination "evidence\runs\siglip2_lora_matched_head_development_v2_bf16"
    Copy-DirectoryToStage -Source $FinalConfirmationRun -RelativeDestination "evidence\runs\fashionpedia_neckline_final_confirmation_v2"

    $ConfigRoot = Join-Path $RepoRoot "configs"
    $ScriptRoot = Join-Path $RepoRoot "scripts"
    Assert-PathExists -Path $ConfigRoot -Label "config source directory"
    Assert-PathExists -Path $ScriptRoot -Label "script source directory"

    Copy-DirectoryToStage -Source $ConfigRoot -RelativeDestination "source_snapshot\configs"
    Copy-DirectoryToStage -Source $ScriptRoot -RelativeDestination "source_snapshot\scripts"

    $DriverFiles = @(
        Get-ChildItem -LiteralPath $RepoRoot -File -Filter "run_amazon07_fashionpedia_*.ps1" -ErrorAction SilentlyContinue |
        Sort-Object Name
    )

    if ($DriverFiles.Count -eq 0) {
        throw "No Amazon07 Fashionpedia PowerShell drivers were found in repository root."
    }

    foreach ($DriverFile in $DriverFiles) {
        Copy-FileToStage -Source $DriverFile.FullName -RelativeDestination (Join-Path "source_snapshot\drivers" $DriverFile.Name)
    }

    $RequirementFiles = @(
        Get-ChildItem -LiteralPath $RepoRoot -File -Filter "requirements*.txt" -ErrorAction SilentlyContinue |
        Sort-Object Name
    )

    if ($RequirementFiles.Count -eq 0) {
        throw "No requirements*.txt files were found in repository root."
    }

    foreach ($RequirementFile in $RequirementFiles) {
        Copy-FileToStage -Source $RequirementFile.FullName -RelativeDestination (Join-Path "source_snapshot\requirements" $RequirementFile.Name)
    }

    $GitState = Get-OptionalGitState -RepositoryRoot $RepoRoot

    $DatasetAndModelDependencies = @"
# Data and model dependencies intentionally excluded from this evidence bundle

This archive preserves code, task definitions, manifests, outputs, selected checkpoints, and hashes. It deliberately excludes raw Fashionpedia images, raw Fashionpedia JSON annotations, and Hugging Face base-model cache files.

## Required external assets for replay

- Fashionpedia train annotation: `instances_attributes_train2020.json`
- Fashionpedia validation annotation: `instances_attributes_val2020.json`
- Fashionpedia train image root: `train\\train`
- Fashionpedia validation image root: `val_test\\test`
- Base model: `google/siglip2-base-patch16-224`
- Python environment: use the included requirements lock files.

## Checkpoint loading boundary

The preserved LoRA adapter is not a standalone SigLIP2 base model. Load the documented base model first, then apply the saved PEFT adapter and its matched 7-class head checkpoint. PEFT adapter artifacts and the original base model are intentionally stored separately.

## Release boundary

This is a local evidence freeze. It has not run hosted CI or been published as a GitHub Release.
"@

    $DatasetDependenciesPath = Join-Path $ReleaseDirectory "DATA_AND_MODEL_DEPENDENCIES.md"
    [System.IO.File]::WriteAllText(
        $DatasetDependenciesPath,
        $DatasetAndModelDependencies,
        [System.Text.UTF8Encoding]::new($false)
    )

    $ReleaseNotes = @"
# Fashionpedia Neckline Attribute Reliability v1.0.0 Local Evidence Freeze

## Qualification status

**LOCAL_EVIDENCE_FROZEN**

This bundle freezes the evidence from a completed, untouched official Fashionpedia validation confirmation. It is not yet a hosted-CI-qualified GitHub Release.

## Fixed task scope

- Dataset: Fashionpedia
- Target category: `33`, `neckline`
- Labels: 7 fine-grained neckline attributes
- Eligibility: category `33`, exactly one target attribute, and positive bbox width and height
- Source task reproduction: 20,800 train/dev annotation-attribute pairs
- Official validation cohort: 654 eligible instances across 644 images
- Source train/dev image overlap: 0

## Official validation result

| Metric | Frozen SigLIP2 + matched head | SigLIP2 vision-attention LoRA + same head | LoRA minus frozen |
|---|---:|---:|---:|
| Macro-F1 | $([math]::Round([double]$FinalSummary.frozen_head.raw_metrics.macro_f1, 6)) | $([math]::Round([double]$FinalSummary.lora.raw_metrics.macro_f1, 6)) | +$([math]::Round([double]$FinalSummary.lora_minus_frozen_head.raw.macro_f1 * 100.0, 2)) pp |
| Raw top-label ECE | $([math]::Round([double]$FinalSummary.frozen_head.raw_metrics.top_label_ece, 6)) | $([math]::Round([double]$FinalSummary.lora.raw_metrics.top_label_ece, 6)) | $([math]::Round(([double]$FinalSummary.lora.raw_metrics.top_label_ece - [double]$FinalSummary.frozen_head.raw_metrics.top_label_ece), 6)) |
| Temperature-calibrated ECE | $([math]::Round([double]$FinalSummary.frozen_head.temperature_calibrated_metrics.top_label_ece, 6)) | $([math]::Round([double]$FinalSummary.lora.temperature_calibrated_metrics.top_label_ece, 6)) | $([math]::Round(([double]$FinalSummary.lora.temperature_calibrated_metrics.top_label_ece - [double]$FinalSummary.frozen_head.temperature_calibrated_metrics.top_label_ece), 6)) |

## Permitted claim

On the official, image-disjoint Fashionpedia validation cohort, vision-attention LoRA improved Macro-F1 from $([math]::Round([double]$FinalSummary.frozen_head.raw_metrics.macro_f1, 3)) to $([math]::Round([double]$FinalSummary.lora.raw_metrics.macro_f1, 3)) (+$([math]::Round([double]$FinalSummary.lora_minus_frozen_head.raw.macro_f1 * 100.0, 2)) percentage points) versus a matched frozen-encoder control.

## Claim boundaries

- Instance-crop, seven-class neckline verification only.
- Scope is Fashionpedia category 33, neckline, under the immutable eligibility rule.
- This is not a full Fashionpedia benchmark, consumer-to-shop retrieval system, or production deployment.
- Raw ECE improved for LoRA. After development-fit temperature scaling, calibrated ECE is slightly lower for the frozen control, so no post-calibration LoRA-superiority claim is made.
- No post-validation checkpoint, hyperparameter, crop, rank, or target-module changes are represented in this evidence bundle.

## Frozen checkpoint selection

- Frozen matched-head checkpoint: development epoch $($FinalSummary.frozen_head.selected_development_epoch)
- LoRA checkpoint: development epoch $($FinalSummary.lora.selected_development_epoch)
- LoRA scope: 48 audited vision encoder self-attention projections
- LoRA adapter parameters: 589,824
- Matched classification-head parameters: 5,383

## Contents

- `evidence/`: manifests, audits, baseline outputs, development outputs, official-validation outputs, selected checkpoints, and final confirmation artifacts.
- `source_snapshot/`: configs, runners, drivers, and dependency locks captured at freeze time.
- `release_evidence_manifest.json`: file-by-file SHA-256 manifest.
- `DATA_AND_MODEL_DEPENDENCIES.md`: intentionally excluded external assets and replay prerequisites.
"@

    $ReleaseNotesPath = Join-Path $ReleaseDirectory "RELEASE_NOTES.md"
    [System.IO.File]::WriteAllText(
        $ReleaseNotesPath,
        $ReleaseNotes,
        [System.Text.UTF8Encoding]::new($false)
    )

    $SourceArtifacts = @(
        [PSCustomObject]@{ label = "forensics"; path = $ForensicsJson },
        [PSCustomObject]@{ label = "task_audit"; path = $TaskAuditJson },
        [PSCustomObject]@{ label = "task_config"; path = $TaskConfigPath },
        [PSCustomObject]@{ label = "development_summary"; path = $DevelopmentSummaryPath },
        [PSCustomObject]@{ label = "final_confirmation_summary"; path = $FinalSummaryPath },
        [PSCustomObject]@{ label = "official_validation_manifest_summary"; path = $FinalManifestSummaryPath },
        [PSCustomObject]@{ label = "lora_adapter_config"; path = $AdapterConfigPath },
        [PSCustomObject]@{ label = "lora_adapter_weights"; path = $AdapterWeights[0].FullName },
        [PSCustomObject]@{ label = "lora_head_checkpoint"; path = $LoRAHeadPath },
        [PSCustomObject]@{ label = "frozen_head_checkpoint"; path = $FrozenHeadPath }
    ) | ForEach-Object {
        [PSCustomObject]@{
            label = $_.label
            source_path = $_.path
            sha256 = (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = [int64](Get-Item -LiteralPath $_.path).Length
        }
    }

    $ReleaseManifest = [ordered]@{
        status = "PASS"
        schema_version = "fashionpedia_neckline_attribute_reliability_local_evidence_freeze_v1"
        release_name = $ReleaseName
        release_status = "LOCAL_EVIDENCE_FROZEN"
        created_utc = (Get-Date).ToUniversalTime().ToString("o")
        no_training_or_inference_executed = $true
        git_state = $GitState
        official_confirmation = [ordered]@{
            evaluation_scope = $FinalSummary.evaluation_scope
            selection_scope = $FinalSummary.selection_scope
            target_category_id = $FinalSummary.data.target_category_id
            target_category_name = $FinalSummary.data.target_category_name
            official_validation_samples = $FinalSummary.data.official_validation_samples
            official_validation_images = $FinalSummary.data.official_validation_images
            source_train_dev_image_overlap_count = $FinalManifestSummary.official_validation.source_train_dev_image_id_overlap_count
            frozen_selected_development_epoch = $FinalSummary.frozen_head.selected_development_epoch
            lora_selected_development_epoch = $FinalSummary.lora.selected_development_epoch
            frozen_raw_macro_f1 = [double]$FinalSummary.frozen_head.raw_metrics.macro_f1
            lora_raw_macro_f1 = [double]$FinalSummary.lora.raw_metrics.macro_f1
            lora_minus_frozen_raw_macro_f1 = [double]$FinalSummary.lora_minus_frozen_head.raw.macro_f1
            frozen_raw_ece = [double]$FinalSummary.frozen_head.raw_metrics.top_label_ece
            lora_raw_ece = [double]$FinalSummary.lora.raw_metrics.top_label_ece
            frozen_calibrated_ece = [double]$FinalSummary.frozen_head.temperature_calibrated_metrics.top_label_ece
            lora_calibrated_ece = [double]$FinalSummary.lora.temperature_calibrated_metrics.top_label_ece
        }
        model = [ordered]@{
            base_model_id = $FinalSummary.model_id
            lora_target_module_count = 48
            lora_adapter_trainable_parameters = 589824
            matched_head_trainable_parameters = 5383
        }
        source_artifact_hashes = @($SourceArtifacts)
        intentional_exclusions = @(
            "Fashionpedia raw images",
            "Fashionpedia raw JSON annotations",
            "Hugging Face base-model cache",
            "Python virtual environment",
            "Git history"
        )
        claim_boundary = $FinalSummary.claim_boundary
    }

    $ManifestPath = Join-Path $ReleaseDirectory "release_evidence_manifest.json"
    [System.IO.File]::WriteAllText(
        $ManifestPath,
        ($ReleaseManifest | ConvertTo-Json -Depth 30),
        [System.Text.UTF8Encoding]::new($false)
    )

    $FileManifest = Get-RelativeFileManifest -Root $ReleaseDirectory
    $FileManifestPath = Join-Path $ReleaseDirectory "release_file_manifest.json"
    [System.IO.File]::WriteAllText(
        $FileManifestPath,
        (@{
            schema_version = "sha256_file_manifest_v1"
            root = $ReleaseName
            file_count = $FileManifest.Count
            files = $FileManifest
        } | ConvertTo-Json -Depth 30),
        [System.Text.UTF8Encoding]::new($false)
    )

    Compress-Archive `
        -Path (Join-Path $ReleaseDirectory "*") `
        -DestinationPath $ReleaseZip `
        -CompressionLevel Optimal `
        -Force

    if (-not (Test-Path -LiteralPath $ReleaseZip)) {
        throw "Release ZIP was not created: $ReleaseZip"
    }

    $ZipHash = (
        Get-FileHash -LiteralPath $ReleaseZip -Algorithm SHA256
    ).Hash.ToLowerInvariant()

    [System.IO.File]::WriteAllText(
        $ReleaseZipSha,
        "$ZipHash  $([System.IO.Path]::GetFileName($ReleaseZip))`n",
        [System.Text.UTF8Encoding]::new($false)
    )

    Copy-Item -LiteralPath $ManifestPath -Destination $ReleaseManifestSidecar -Force
    Copy-Item -LiteralPath $ReleaseNotesPath -Destination $ReleaseNotesSidecar -Force

    $ZipItem = Get-Item -LiteralPath $ReleaseZip
    $ReleaseFileCount = @(
        Get-ChildItem -LiteralPath $ReleaseDirectory -Recurse -File -Force
    ).Count

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "POST-VALIDATION EVIDENCE FREEZE QUALIFIED"
    Write-Host "============================================================"
    Write-Host "STATUS: PASS"
    Write-Host "RELEASE_STATUS: LOCAL_EVIDENCE_FROZEN"
    Write-Host "OFFICIAL_VALIDATION_SAMPLES: $($FinalSummary.data.official_validation_samples)"
    Write-Host "SOURCE_VALIDATION_IMAGE_OVERLAP: $($FinalManifestSummary.official_validation.source_train_dev_image_id_overlap_count)"
    Write-Host "FROZEN_RAW_MACRO_F1: $([math]::Round([double]$FinalSummary.frozen_head.raw_metrics.macro_f1, 6))"
    Write-Host "LORA_RAW_MACRO_F1: $([math]::Round([double]$FinalSummary.lora.raw_metrics.macro_f1, 6))"
    Write-Host "LORA_MINUS_FROZEN_RAW_MACRO_F1: $([math]::Round([double]$FinalSummary.lora_minus_frozen_head.raw.macro_f1, 6))"
    Write-Host "RELEASE_DIRECTORY: $ReleaseDirectory"
    Write-Host "RELEASE_ZIP: $ReleaseZip"
    Write-Host "RELEASE_ZIP_SHA256: $ZipHash"
    Write-Host "RELEASE_ZIP_SHA256_SIDECAR: $ReleaseZipSha"
    Write-Host "RELEASE_MANIFEST: $ReleaseManifestSidecar"
    Write-Host "RELEASE_NOTES: $ReleaseNotesSidecar"
    Write-Host "RELEASE_FILE_COUNT: $ReleaseFileCount"
}
