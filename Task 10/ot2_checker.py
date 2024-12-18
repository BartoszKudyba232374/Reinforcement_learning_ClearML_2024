from stable_baselines3.common.env_checker import check_env
from ot2_env_wrapper import WrappedEnv

env = WrappedEnv(render=False, max_step=1000)
env.reset()

check_env(env)
