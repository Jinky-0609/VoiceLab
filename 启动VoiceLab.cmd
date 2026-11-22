@echo off
chcp 65001 >nul
title VoiceLab
cd /d "%~dp0"

echo.
echo   VoiceLab 正在启动...
echo.

if not exist ".venv-tts\Scripts\python.exe" (
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

echo   浏览器会自动打开 http://127.0.0.1:8765
echo   关闭本窗口即停止服务。
echo.
echo  ==========================================
echo.

".venv-tts\Scripts\python.exe" tools\webui.py

echo.
echo   服务已停止。
pause
