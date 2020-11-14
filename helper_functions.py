import sys
import cv2
import numpy as np
from collections import deque
import random
import gym
from gym import spaces
# from gym.wrappers.monitor import Monitor
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import tensorflow as tf

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def get_size(data_struct):
	return sys.getsizeof(data_struct)

def create_env(env_name):
    env = gym.make(env_name)
    # env = AtariPreprocessing(env=env, grayscale_newaxis=True, scale_obs=True)
    # env = FrameStack(env, 4)
    # env = FireResetEnv(env)
    return env

def get_random_num(min, max, factor=1):
	return random.randrange(min, max) * factor

def show_image(*args):
	i = 0
	for img in args:
		cv2.imshow('Image{}'.format(i), img)
		i += 1
	cv2.waitKey()
	cv2.destroyAllWindows()

def show_progress(frame_count, frames_per_epoch, loss):
	fill = '='
	length = 50
	fill_len = int(frame_count * length // frames_per_epoch)
	arrow = '' if frame_count == frames_per_epoch else '>'
	bar = fill * fill_len + arrow + '-' * (length - fill_len)
	print(f'\r[{bar}] {frame_count}/{frames_per_epoch}  loss={loss}', end='\r')
	if frame_count == frames_per_epoch:
		print()
		return True
	return False
import time
def run_demo(env_name, model):
    env = create_env(env_name)
    state = np.array(env.reset())
    last_state = state
    print(env.action_space.n)
    total = 0
    no_change_count = 0
    for _ in range(10000):
        env.render(mode='human')
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        pred = model(state_tensor, training=False)
        action = np.argmax(pred)
        state, reward, done, info = env.step(action)
        state = np.array(state)
        temp = np.array_equal(last_state, state)
        # print(temp)
        if temp:
            no_change_count += 1
        last_state = state
        total += reward
        if no_change_count > 100:
            break
        if done:
            break
        # time.sleep(0.05)
    print('Demo reward: {}'.format(total))
    print('no_change_count: {}'.format(no_change_count))
    env.close()
