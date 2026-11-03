"""财报文本分析助手（本地版）入口 —— 双击 启动.bat，或 streamlit run app.py

界面代码在 ui.py。Windows 多进程（spawn）的子进程会以 __mp_main__ 身份重新导入本文件，
所以这里只在 Streamlit 真正运行脚本时（__name__ == "__main__"）才执行界面，避免子进程重跑整个页面。
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("ui.py")), run_name="ui")
