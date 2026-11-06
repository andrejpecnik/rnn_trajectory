@echo off
REM AI vs AI: modry tank mieri pomocou GRU, cerveny pomocou konstantnej rychlosti
cd /d "%~dp0"
".venv\Scripts\python.exe" -m tankgame watch %*
