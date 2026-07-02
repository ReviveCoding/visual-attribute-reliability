@echo off
setlocal DisableDelayedExpansion
cd /d "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -u "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\scripts\audit_siglip2_vision_lora_targets.py" ^
  --model-id "google/siglip2-base-patch16-224" ^
  --cache-dir "C:\Users\bjw-0\ML_Cache\huggingface" ^
  --output-json "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\siglip2_vision_lora_target_audit.json" ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_target_audit\siglip2_vision_lora_target_audit.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_target_audit\siglip2_vision_lora_target_audit.stderr.log"

exit /b %ERRORLEVEL%