# Fashionpedia Neckline Attribute Reliability: CI Evidence Fixture

This fixture is a **small, tracked, static subset** of the local evidence freeze. It is for hosted CI contract tests only and is not a distributable model package or a replacement for the local release evidence bundle.

## Confirmed scope

- Fashionpedia category `33`, `neckline`
- Seven-class instance-crop neckline verification
- Official validation subset eligible under the immutable task rule
- 654 eligible instances across 644 images
- Zero source-train/dev to validation image overlap

## Frozen comparison

- Matched frozen SigLIP2 control Macro-F1: `0.5797703254640079`
- Vision-attention LoRA Macro-F1: `0.6681089306597886`
- LoRA uplift: `+8.83 percentage points` (`+0.08833860519578074` absolute Macro-F1)

## Claim boundaries

- This is **not a full Fashionpedia benchmark**.
- This is not consumer-to-shop retrieval or a production fashion-catalog system.
- The evidence supports a category-scoped, seven-class neckline-verification claim only.
- Raw top-label ECE favored LoRA in the frozen final-confirmation evidence, but there is **no post-calibration LoRA-superiority claim**.
- Full checkpoint-presence, full release-file SHA-256 verification, and source-artifact hash verification remain local evidence-release gates, not hosted-CI fixture checks.
