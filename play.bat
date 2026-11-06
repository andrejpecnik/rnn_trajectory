@echo off
REM Spusti hru proti RNN nepriatelovi. Dalsie parametre: --difficulty easy^|normal^|hard, --enemy cv, --record
cd /d "%~dp0"
".venv\Scripts\python.exe" -m tankgame play %*
