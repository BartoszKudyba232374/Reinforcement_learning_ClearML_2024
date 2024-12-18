import argparse
import os
import wandb
from ot2_env_wrapper import WrappedEnv
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from clearml import Task


# Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# It can also be helpful to include the hyperparameters in the task name

# task = Task.init(project_name='Mentor Group S/Group 1', task_name='Experiment1')
task = Task.init(project_name='Mentor Group S/Bartosz Kudyba', task_name='Experiment-test')
# copy these lines exactly as they are
# setting the base docker image

# copy these lines exactly as they are
# setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
# setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

os.environ['WANDB_API_KEY'] = '47f4fed852265faa4acb02db524138518171f840'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size_multiple", type=float, default=0.05)
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

config = dict(
    total_timesteps=100000,
    policy='MlpPolicy',
    n_steps_max=1000,
    device='cpu',
    save_freq=10000,
    learning_rate=args.learning_rate,
    batch_size=int(args.n_steps*args.batch_size_multiple),
    n_steps=args.n_steps,
    n_epochs=args.n_epochs
)

run = wandb.init(project='OT2_RL_test',
                 config=config,
                 sync_tensorboard=True)

env = WrappedEnv(render=False,
                 max_step=config['n_steps_max'])

model = PPO(policy='MlpPolicy',
            env=env,
            device=config['device'],
            tensorboard_log=f'runs/Rl_test_{run.id}',
            learning_rate=args.learning_rate,
            batch_size=int(args.n_steps*args.batch_size_multiple),
            n_steps=args.n_steps,
            n_epochs=args.n_epochs)

cb = WandbCallback(model_save_freq=config['save_freq'],
                   model_save_path=f'models/RL_test/model_{run.id}')


for i in range(config['total_timesteps']//config['save_freq']):
    model.learn(total_timesteps=config['save_freq'],
                progress_bar=True,
                callback=cb,
                tb_log_name=f'runs/Rl_test_{run.id}',
                reset_num_timesteps=False)
    model.save(f"models/RL_test/run_{run.id}/step_{config['save_freq']*(i+1)}")


test_env = WrappedEnv(render=True, max_step=config['n_steps_max'])

for episode in range(10):
    obs, _ = test_env.reset()
    done = False
    step = 0

    while not done:
        # Take a random action from the environment's action space
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        done = truncated or terminated
        if done:
            print(f"Episode finished after {step} steps. Info: {info}")
            break
