@echo off
setlocal
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -u "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\scripts\probe_siglip2_image_processor.py" ^
  --model-id "google/siglip2-base-patch16-224" ^
  --cache-dir "C:\Users\bjw-0\ML_Cache\huggingface" ^
  --output-json "C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\siglip2_image_processor_smoke_test.json" ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_processor_smoke_test\siglip2_image_processor_smoke_test.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_processor_smoke_test\siglip2_image_processor_smoke_test.stderr.log"

exit /b %ERRORLEVEL%