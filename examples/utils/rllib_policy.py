import copy
import pickle5 as pkl
import threading
from pathlib import Path

import os
import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.agents.qmix.qmix import QMixTorchPolicy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces import space_utils

import mate
from mate.agents.base import AgentBase


__all__ = [
    'load_checkpoint',
    'get_preprocessor',
    'get_space_flat_size',
    'DEFAULT_POLICY_ID',
    'default_policy_mapping_fn',
    'SHARED_POLICY_ID',
    'shared_policy_mapping_fn',
    'independent_policy_mapping_fn',
    'RLlibPolicyMixIn',
    'RLlibGroupedPolicyMixIn',
]


SHARED_POLICY_ID = 'shared_policy'

_CHECKPOINT_CACHE_LOCK = threading.RLock()
_CHECKPOINT_CACHE = {}

# [superboySB] I met bug: AttributeError: 'PosixPath' object has no attribute 'readlink'
# def load_checkpoint(path):
#     if path is not None:
#         path = Path(path).absolute()
#         
#         # try:
#         #     path = path.readlink()
#         # except OSError:
#         #     pass
#         path = Path(os.readlink(path))
#         # print(path)
#         with _CHECKPOINT_CACHE_LOCK:
#             try:
#                 params, worker = _CHECKPOINT_CACHE[path]
#             except KeyError:
#                 with (path.parent.parent / 'params.pkl').open(mode='rb') as file:
#                     params = pkl.load(file)
#                 with path.open(mode='rb') as file:
#                     checkpoint = pkl.load(file)
#                     worker = pkl.loads(checkpoint['worker'])
#                 _CHECKPOINT_CACHE[path] = (params, worker)
#     else:
#         params = None
#         worker = None

#     return path, worker, params


def load_checkpoint(path):
    if path is not None:
        path = Path(path).absolute()
        
        # 调试信息: 打印原始路径
        print(f'Original path: {path}')

        # 确保路径存在
        if not path.exists():
            raise FileNotFoundError(f'Path does not exist: {path}')
        
        # 尝试读取符号链接，如果不是符号链接则使用原路径
        try:
            if os.path.islink(path):
                resolved_path = Path(os.readlink(path))
                print(f'Resolved symbolic link: {resolved_path}')
            else:
                resolved_path = path
                print(f'Path is not a symbolic link, using original path: {resolved_path}')
        except OSError as e:
            print(f'Error occurred while reading symbolic link: {e}')
            raise
        
        with _CHECKPOINT_CACHE_LOCK:
            try:
                params, worker = _CHECKPOINT_CACHE[resolved_path]
            except KeyError:
                with (resolved_path.parent.parent / 'params.pkl').open(mode='rb') as file:
                    params = pkl.load(file)
                with resolved_path.open(mode='rb') as file:
                    checkpoint = pkl.load(file)
                    worker = pkl.loads(checkpoint['worker'])
                _CHECKPOINT_CACHE[resolved_path] = (params, worker)
    else:
        params = None
        worker = None

    return resolved_path, worker, params

def get_preprocessor(space):
    return ModelCatalog.get_preprocessor_for_space(space)


def get_space_flat_size(space):
    return get_preprocessor(space).size


def default_policy_mapping_fn(agent_id, **kwargs):
    return DEFAULT_POLICY_ID


def shared_policy_mapping_fn(agent_id, **kwargs):
    return SHARED_POLICY_ID


def independent_policy_mapping_fn(agent_id, **kwargs):
    return agent_id

def print_state_types(state, prefix=''):
    if isinstance(state, dict):
        for key, value in state.items():
            print(f'{prefix}{key}: {type(value)}')
            print_state_types(value, prefix + '  ')
    elif isinstance(state, list):
        for index, item in enumerate(state):
            print(f'{prefix}[{index}]: {type(item)}')
            print_state_types(item, prefix + '  ')
    elif isinstance(state, np.ndarray):
        print(f'{prefix}ndarray dtype: {state.dtype}')
    else:
        print(f'{prefix}{type(state)}')

def convert_ndarray_to_float32(state):
    if isinstance(state, dict):
        for key, value in state.items():
            state[key] = convert_ndarray_to_float32(value)
    elif isinstance(state, list):
        state = [convert_ndarray_to_float32(item) for item in state]
    elif isinstance(state, np.ndarray):
        if state.dtype == np.object:
            state = np.array(state.tolist(), dtype=np.float32)
        else:
            state = state.astype(np.float32)
    return state

def convert_optimizer_variables(optimizer_vars):
    if isinstance(optimizer_vars, list):
        for i in range(len(optimizer_vars)):
            if isinstance(optimizer_vars[i], dict):
                for key, value in optimizer_vars[i].items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, np.ndarray) and v.dtype == np.object:
                                optimizer_vars[i][key][k] = np.array(v.tolist(), dtype=np.float32)
                            elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                                optimizer_vars[i][key][k] = v.astype(np.float32)
    return optimizer_vars

class RLlibPolicyMixIn(AgentBase):
    POLICY_CLASS = PPOTorchPolicy
    DEFAULT_CONFIG = None

    def __init__(self, config, checkpoint_path, make_env, seed=None):
        super().__init__(seed=seed)

        self.checkpoint_path, self.worker, self.params = load_checkpoint(checkpoint_path)

        if config is None:
            config = self.params or self.DEFAULT_CONFIG
        self.config = copy.deepcopy(config)

        self.need_convert_coordinates = False
        self.need_rescale_observation = False
        self.hidden_state = None

        self.preprocessor = None
        self.make_env = make_env

        env_config = self.config.get('env_config', {})

        # Reduce unnecessary RLlib policy creation (use light-weight rule-based agent)
        if 'opponent_agent_factory' in env_config:
            env_config['opponent_agent_factory'] = {
                mate.Team.CAMERA: mate.RandomTargetAgent,
                mate.Team.TARGET: mate.RandomCameraAgent,
            }[self.TEAM]
        if 'camera_agent_factory' in env_config:
            env_config['camera_agent_factory'] = mate.RandomCameraAgent
        if 'target_agent_factory' in env_config:
            env_config['target_agent_factory'] = mate.RandomTargetAgent

        self.policy = self.get_policy()

        multiagent = self.config.get('multiagent', {})
        self.policy_mapping_fn = multiagent.get('policy_mapping_fn', default_policy_mapping_fn)
        self.policy_id = None

        self.unsquash_action = self.config.get('normalize_actions', True)
        self.clip_action = self.config.get('clip_actions', False)

        if self.worker is not None:
            self.load_policy_state()

    def get_policy(self):
        with self.make_env(self.config.get('env_config', {})) as dummy_env:
            mate.wrappers.typing.assert_mate_environment(dummy_env)
            self.need_convert_coordinates = isinstance(dummy_env, mate.RelativeCoordinates)
            self.need_rescale_observation = isinstance(dummy_env, mate.RescaledObservation)

            self.preprocessor = get_preprocessor(dummy_env.observation_space)

            policy = self.POLICY_CLASS(
                self.preprocessor.observation_space,
                dummy_env.action_space,
                config=dict(self.config, num_gpus=0, num_gpus_per_worker=0),
            )

        return policy

    def clone(self):
        return self.__class__(
            config=self.config,
            checkpoint_path=self.checkpoint_path,
            make_env=self.make_env,
            seed=self.np_random.randint(np.iinfo(int).max),
        )

    def reset(self, observation):
        super().reset(observation)

        if self.worker is not None:
            self.load_policy_state()

        self.hidden_state = self.policy.get_initial_state()

    def preprocess_raw_observation(self, observation):
        preprocessed_observation = observation
        if self.need_convert_coordinates:
            preprocessed_observation = self.convert_coordinates(preprocessed_observation)
        if self.need_rescale_observation:
            preprocessed_observation = self.rescale_observation(preprocessed_observation)

        return preprocessed_observation

    def preprocess_observation(self, observation):
        preprocessed_observation = self.preprocess_raw_observation(observation)

        if not isinstance(self.preprocessor, NoPreprocessor):
            dummy_preprocessed_observation = np.zeros(
                shape=self.preprocessor.observation_space.shape,
                dtype=self.preprocessor.observation_space.dtype,
            )
            dummy_preprocessed_observation[
                : preprocessed_observation.size
            ] = preprocessed_observation.ravel()
            return dummy_preprocessed_observation
        return preprocessed_observation.ravel()

    def compute_single_action(self, observation, state, info=None, deterministic=None):
        preprocessed_observation = self.preprocess_observation(observation)

        explore = not deterministic if deterministic is not None else None

        results = self.policy.compute_single_action(
            preprocessed_observation, state=state, info=info, explore=explore
        )
        action, state, *_ = results

        if self.unsquash_action:
            action = space_utils.unsquash_action(action, self.policy.action_space_struct)
        elif self.clip_action:
            action = space_utils.clip_action(action, self.policy.action_space_struct)

        return action, state

    @property
    def model(self):
        return self.policy.model

    def load_policy_state(self, agent_id=None):
        assert self.worker is not None

        agent_id = agent_id or self.agent_id

        policy_id = self.policy_mapping_fn(agent_id)
        policy_id = policy_id if policy_id in self.worker['state'] else DEFAULT_POLICY_ID

        if self.policy_id is None or policy_id != self.policy_id:
            self.policy_id = policy_id

            # 获取当前 policy 的状态
            state = self.worker['state'][self.policy_id]

            # 打印状态的类型
            # print('Before conversion:')
            # print_state_types(state)

            # 递归转换状态字典中的所有 np.ndarray 类型
            # state = convert_ndarray_to_float32(state)

            # 专门处理 _optimizer_variables
            # if '_optimizer_variables' in state:
            #     state['_optimizer_variables'] = convert_optimizer_variables(state['_optimizer_variables'])

            # 再次打印转换后的状态类型
            # print('After conversion:')
            # print_state_types(state)

            self.policy.set_state(state)

    def __reduce__(self):
        return self.__class__, (
            self.config,
            self.checkpoint_path,
            self.make_env,
            self.np_random.randint(np.iinfo(int).max),
        )


class RLlibGroupedPolicyMixIn(RLlibPolicyMixIn):
    POLICY_CLASS = QMixTorchPolicy

    def get_policy(self):
        with self.make_env(self.config.get('env_config', {})) as dummy_env:
            mate.wrappers.typing.assert_multi_agent_environment(dummy_env.env)
            self.need_convert_coordinates = isinstance(dummy_env.env, mate.RelativeCoordinates)
            self.need_rescale_observation = isinstance(dummy_env.env, mate.RescaledObservation)

            self.grouped_observation_space = dummy_env.observation_space
            self.preprocessor = get_preprocessor(self.grouped_observation_space)

            policy = self.POLICY_CLASS(
                dummy_env.observation_space,
                dummy_env.action_space,
                config=dict(self.config, num_gpus=0, num_gpus_per_worker=0),
            )

        return policy

    def compute_single_action(self, observation, state, info=None, deterministic=None):
        dummy_joint_action, state = super().compute_single_action(
            observation, state=state, info=info, deterministic=deterministic
        )

        action = dummy_joint_action[0]
        return action, state
