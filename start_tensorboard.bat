call C:\Users\balasubramanian\Anaconda3\Scripts\activate.bat
call conda activate safir_II
call start  firefox -new-tab http://localhost:6006/
@echo off
setlocal

set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please choose a folder.',0,'E:\Lakshman_SAFIR_II\SCENARIO-NET\runs\').self.path""

for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "folder=%%I"

setlocal enabledelayedexpansion
call tensorboard --logdir=!folder! 