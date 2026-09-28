@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Reconhecimento de Placas
echo ============================================
echo  Reconhecimento de Placas - app desktop
echo ============================================
echo=
py --version >nul 2>&1
if not errorlevel 1 goto deps
echo [ERRO] Python nao encontrado. Instale em https://www.python.org/downloads/
echo        Marque a opcao "Add python.exe to PATH" na instalacao.
pause
exit /b 1
:deps
py -c "import cv2, numpy, PIL, requests, tkinter" >nul 2>&1
if not errorlevel 1 goto rodar
echo Instalando dependencias pela primeira vez, aguarde...
echo Isso pode demorar alguns minutos na primeira vez.
py -m pip install -r requirements-app.txt
echo=
:rodar
echo Abrindo o Reconhecimento de Placas...
echo Para encerrar, feche a janela do programa.
echo=
py app_gui.py
pause
