#!/bin/bash

sudo apt update
# sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg
sudo apt install -y libgl1-mesa-glx

cd ~/projects/
if [ -d "Image_upscaling" ]; then
    cd Image_upscaling && git pull origin main
else
    git clone https://github.com/harikp196/Image_upscaling.git
fi

cd ~/projects
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r ~/projects/Image_upscaling/requirements.txt

nohup python3 ~/projects/Image_upscaling/app.py > ~/projects/project1.log 2>&1 &