#!/bin/bash

# run in project folder
mkdir -p logs

# install python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13
sudo apt install python3.13-venv

# make virtual environment / install python packages
# --copies: install a real python binary (not a symlink) so setcap below
# elevates only this venv's interpreter, not the system-wide one.
/usr/bin/python3.13 -m venv --copies .venv
source .venv/bin/activate

pip install -r requirements.txt

# Grant the venv python the two capabilities the dyno stack needs, so it no
# longer has to run under sudo:
#   cap_net_raw  - pysoem's raw EtherCAT socket
#   cap_sys_nice - SCHED_FIFO real-time scheduling for the cyclic loop
# Re-run this if the venv is ever rebuilt.
sudo setcap cap_net_raw,cap_sys_nice+ep .venv/bin/python3.13

# install pyqt dependencies
sudo apt install libxcb-cursor0

# install tkinter
sudo apt-get install python3.13-tk