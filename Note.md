# 自博弈实现方式的调研
因为各种原因我只能用python3.7
```sh
# （Optional）required on Windows 建议测试 
git config --global core.symlinks true 

# 拉取代码 
git clone https://github.com/superboySB/mate.git

# 暂时还是当做一台机器训练与推理来处理
# (但目前感觉python3.7只能用老版nashpy)
conda create -n mate python=3.7 && conda activate mate
conda install libglu
cd mate && pip install -r requirements.txt && pip install -e . 
```
运行PVE的demo
```sh
bash scripts/camera.mappo.sh
```
运行psro的demo
```sh
bash scripts/psro.sh
python3 -m examples.psro.train
```
