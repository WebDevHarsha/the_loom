@echo off
rem Portable TinyTeX install - everything stays inside this project dir (F:\RB).
rem No admin or system permissions needed; installs to <project>\TinyTeX only.
setlocal
set "TINYTEX_ZIP=%~dp0..\TinyTeX-1-windows.exe"
set "TINYTEX_DIR=%~dp0..\TinyTeX"

if exist "%TINYTEX_DIR%\bin\windows\pdflatex.exe" (
    echo TinyTeX already installed at %TINYTEX_DIR%
    exit /b 0
)

echo Downloading TinyTeX-1 for Windows (~74 MB)...
rem Version-pinned URL; update the tag below when TinyTeX releases a newer one.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri 'https://github.com/rstudio/tinytex-releases/releases/download/v2026.11/TinyTeX-1-windows-v2026.11.exe' -OutFile '%TINYTEX_ZIP%'"
if errorlevel 1 (
    echo ERROR: download failed.
    exit /b 1
)

echo Extracting to %TINYTEX_DIR% ...
"%TINYTEX_ZIP%" x -o"%TINYTEX_DIR%" -y>nul
if errorlevel 1 (
    echo ERROR: extraction failed.
    exit /b 1
)

rem The archive extracts as TinyTeX\TinyTeX\... - flatten it.
if exist "%TINYTEX_DIR%\TinyTeX" (
    move /y "%TINYTEX_DIR%\TinyTeX\*" "%TINYTEX_DIR%">nul 2>&1
    rmdir /s /q "%TINYTEX_DIR%\TinyTeX"
)

del /q "%TINYTEX_ZIP%"
echo Done. TinyTeX installed at %TINYTEX_DIR%
