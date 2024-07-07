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
有一些源码的问题需要处理,参考[issue](https://github.com/ray-project/ray/issues/26557)，首先是`/home/ps/miniconda3/envs/mate/lib/python3.7/site-packages/ray/rllib/policy/torch_policy.py`这里
```py
@override(Policy)
    @DeveloperAPI
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):
                # TODO: Fix
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = False if v["momentum_buffer"] is None else v["momentum_buffer"]
                # TODO: End fix
                        
                optim_state_dict = convert_to_torch_tensor(s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)
```

运行PVE的demo，效果挺不错的
```sh
bash scripts/camera.mappo.sh
```
测试可以参考下面的跑法，详见相应的`__main__.py`：
```sh
python3 -m mate.evaluate --episodes 1 --render-communication         --camera-agent examples.mappo:MAPPOCameraAgent         --camera-kwargs '{ "checkpoint_path": "examples/mappo/camera/ray_results/MAPPO/latest-checkpoint" }' --no-render
```
运行psro的demo，等一波结果
```sh
bash scripts/psro.sh
```
