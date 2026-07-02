@echo off
setlocal DisableDelayedExpansion
cd /d "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -u "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\scripts\run_siglip2_frozen_neckline_baseline.py" ^
  --train-manifest "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\train_manifest.csv" ^
  --dev-manifest "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\dev_manifest.csv" ^
  --manifest-summary "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\manifests\fashionpedia_neckline_attribute_v1b_group_stratified\manifest_summary.json" ^
  --config "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\configs\siglip2_frozen_neckline_baseline_v1.json" ^
  --run-dir "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\runs\siglip2_frozen_neckline_v1_20260701T190347" ^
  --cache-dir "C:\Users\bjw-0\ML_Cache\huggingface" ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_frozen_neckline_baseline\20260701T190347_siglip2_frozen_neckline.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_frozen_neckline_baseline\20260701T190347_siglip2_frozen_neckline.stderr.log"

exit /b %ERRORLEVEL%