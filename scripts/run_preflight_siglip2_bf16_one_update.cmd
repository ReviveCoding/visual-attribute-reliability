@echo off
setlocal DisableDelayedExpansion
cd /d "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -u "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\scripts\preflight_siglip2_bf16_one_update.py" ^
  --train-manifest "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\train_manifest.csv" ^
  --development-config "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\configs\siglip2_lora_matched_head_development_v1.json" ^
  --lora-audit-json "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\siglip2_vision_lora_target_audit.json" ^
  --model-id "google/siglip2-base-patch16-224" ^
  --cache-dir "C:\Users\bjw-0\ML_Cache\huggingface" ^
  --output-json "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\siglip2_bf16_one_update_preflight.json" ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_bf16_one_update_preflight\siglip2_bf16_one_update_preflight.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_bf16_one_update_preflight\siglip2_bf16_one_update_preflight.stderr.log"

exit /b %ERRORLEVEL%