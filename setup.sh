# uncomment if needed on your machine
# apt-get update && apt-get install nano zip ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ python3-distutils python3-apt -y 

pip install --no-cache-dir -r requirements.txt # No cache dir important for scratch configuration

python3 setup.py build_ext --inplace
