#!/bin/bash

# run in project folder
mkdir -p logs

# install python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13
sudo apt install python3.13-venv

# make virtual environment / install python packages
/usr/bin/python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# install pyqt dependencies
sudo apt install libxcb-cursor0

# install tkinter
sudo apt-get install python3.13-tk