@echo off
title Applio v2 - Improved Version

echo.
echo ========================================
echo   Applio v2 - Enhanced Performance ^& UI
echo ========================================
echo.

REM 현재 디렉토리로 이동
cd /d "%~dp0"

REM Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM 개선된 버전 실행
echo [INFO] Starting improved Applio...
echo.
python app_v2.py %*

pause