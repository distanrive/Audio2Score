@echo off
rem Audio2Score GUI 启动示例（通用，不含个人用户名）。
rem 用法：复制本文件为 run_gui.bat，把 pythonw 路径改成你自己 conda 环境的路径即可。
rem 默认假设 conda 装在用户目录（%USERPROFILE%\.conda）。
cd /d "%~dp0"
start "" "%USERPROFILE%\.conda\envs\audio2score\pythonw.exe" gui.py
