# Local Evidence Release

## Release identity

| Field | Value |
|---|---|
| Release status | `LOCAL_EVIDENCE_FROZEN` |
| Release directory | `dist/fashionpedia-neckline-attribute-reliability-v1.0.0-local-evidence` |
| Archive | `fashionpedia-neckline-attribute-reliability-v1.0.0-local-evidence.zip` |
| Archive SHA-256 | `00d2e1b056c41373fff32d27471a72d9fe76bf94cde5c53598595fe145853b03` |
| Release file count | 125 |
| Official validation instances | 654 |
| Source-validation image overlap | 0 |
| Frozen raw Macro-F1 | 0.579770 |
| LoRA raw Macro-F1 | 0.668109 |
| LoRA minus frozen Macro-F1 | +0.088339 |

The archive, sidecar hash, release manifest, and release notes are intentionally excluded from Git history. They remain local artifacts or may be distributed as release assets without placing large binary evidence in the repository history.

## Verify the archive hash

```powershell
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Zip = Join-Path $RepoRoot `
  "dist\fashionpedia-neckline-attribute-reliability-v1.0.0-local-evidence.zip"

(Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
```

The output must equal:

```text
00d2e1b056c41373fff32d27471a72d9fe76bf94cde5c53598595fe145853b03
```

## Verify the canonical local release

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q `
  tests\test_release_evidence_contracts.py `
  -p no:cacheprovider
```

The local evidence suite validates:

- Release semantic contract
- Final-confirmation metrics and selection boundary
- Task configuration
- Source-task reconstruction and image disjointness
- Checkpoint and adapter presence
- Source-artifact SHA-256 values
- Full release-file SHA-256 manifest
- Claim-boundary text

## Claim boundary

The local evidence release is a hash-verified record of this category-scoped study. It is not a general Fashionpedia benchmark, a public inference service, a production deployment, or an independently replicated result.

A local evidence-contract pass verifies the packaged local evidence against its manifest. It does not independently rerun model training or full validation inference.
