@echo off
setlocal DisableDelayedExpansion
cd /d "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -u "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\scripts\run_siglip2_lora_matched_head_development.py" ^
  --train-manifest "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\train_manifest.csv" ^
  --dev-manifest "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\dev_manifest.csv" ^
  --manifest-summary "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\manifest_summary.json" ^
  --hog-config "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\configs\hog_neckline_baseline_v1.json" ^
  --lora-audit-json "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\siglip2_vision_lora_target_audit.json" ^
  --lora-preflight-json "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\siglip2_lora_matched_head_preflight.json" ^
  --config "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\configs\siglip2_lora_matched_head_development_v1.json" ^
  --model-id "google/siglip2-base-patch16-224" ^
  --cache-dir "C:\Users\bjw-0\ML_Cache\huggingface" ^
  --run-dir "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\runs\siglip2_lora_matched_head_development_v1_20260701T193221" ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_matched_head_development\20260701T193221_siglip2_lora_matched_head_development.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_matched_head_development\20260701T193221_siglip2_lora_matched_head_development.stderr.log"

exit /b %ERRORLEVEL%