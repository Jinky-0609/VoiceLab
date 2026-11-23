@echo off
chcp 65001 >nul
title VoiceLab
cd /d "%~dp0"

if not exist ".venv-tts\Scripts\python.exe" (
    echo.
    echo   [错误] 找不到 .venv-tts
    echo   请先运行 install.ps1 安装，或按 docs\环境重建.md 手动部署。
    echo.
    pause
    exit /b 1
)

if not exist "config.json" (
    if exist "config.example.json" (
        copy "config.example.json" "config.json" >nul
        echo   [提示] 已生成 config.json
    )
)

echo.
echo   VoiceLab 启动中...
echo   关闭本窗口即停止服务。
echo.

".venv-tts\Scripts\python.exe" tools\launch.py

echo.
echo   服务已停止。
pause
