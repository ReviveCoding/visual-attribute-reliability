@echo off
setlocal DisableDelayedExpansion
cd /d "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability"
set PYTHONUTF8=1

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -m pip install --isolated --no-input --disable-pip-version-check -r "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\requirements-siglip2-lora.txt" ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_matched_head_preflight\install_peft.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_matched_head_preflight\install_peft.stderr.log"

if errorlevel 1 exit /b %ERRORLEVEL%

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -m pip check ^
  1>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_matched_head_preflight\pip_check.stdout.log" ^
  2>"C:\Users\bjw-0\ML_Outputs\Amazon07_fashionpedia-attribute-reliability\logs\siglip2_lora_matched_head_preflight\pip_check.stderr.log"

if errorlevel 1 exit /b %ERRORLEVEL%

"C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\.venv\Scripts\python.exe" -m pip freeze > "C:\Users\bjw-0\Downloads\Amazon07_fashionpedia-attribute-reliability\requirements-siglip2-lora.lock.txt"

exit /b %ERRORLEVEL%