@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 首次运行：正在创建 Python 环境并安装依赖，需要几分钟 ...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
echo 正在启动，浏览器会自动打开 http://localhost:8501 ；关闭此窗口即退出程序。
".venv\Scripts\python.exe" -m streamlit run app.py
pause
