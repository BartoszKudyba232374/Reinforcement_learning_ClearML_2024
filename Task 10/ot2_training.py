# import argparse
# import os
# import wandb
# from ot2_env_wrapper import WrappedEnv
# from stable_baselines3 import PPO
# from wandb.integration.sb3 import WandbCallback
# from stable_baselines3.common.callbacks import BaseCallback
# import clearml

# # Custom ClearML logging callback
# class ClearMLCallback(BaseCallback):
#     """
#     A custom callback for logging training metrics to ClearML.
#     """
#     def __init__(self, verbose=0):
#         super(ClearMLCallback, self).__init__(verbose)
    
#     def _on_step(self) -> bool:
#         # Log training metrics at each timestep
#         if self.n_calls % config['save_freq'] == 0:  # Log every 1000 timesteps
#             mean_reward = self.locals["infos"][0].get("episode", {}).get("r", 0)
#             total_timesteps = self.num_timesteps
#             episode_length = self.locals["infos"][0].get("episode", {}).get("l", 0)
#             self.logger.record("reward/mean_reward", mean_reward)
#             self.logger.record("time/total_timesteps", total_timesteps)
#             self.logger.record("episode/length", episode_length)
#         return True

# # Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# # It can also be helpful to include the hyperparameters in the task name

# # task = Task.init(project_name='Mentor Group S/Group 1', task_name='Experiment1')
# task = clearml.Task.init(project_name='Mentor Group S/Bartosz Kudyba', task_name='Experiment-test')
# # copy these lines exactly as they are
# # setting the base docker image

# # copy these lines exactly as they are
# # setting the base docker image
# task.set_base_docker('deanis/2023y2b-rl:latest')
# # setting the task to run remotely on the default queue
# task.execute_remotely(queue_name="default")


# os.environ['WANDB_API_KEY'] = '47f4fed852265faa4acb02db524138518171f840'

# parser = argparse.ArgumentParser()
# parser.add_argument("--total_timesteps", type=float, default=1000000)
# parser.add_argument("--learning_rate", type=float, default=0.0003)
# parser.add_argument("--batch_size_multiple", type=float, default=0.05)
# parser.add_argument("--n_steps", type=int, default=1000)
# parser.add_argument("--clip_range", type=float, default=0.2)
# parser.add_argument("--n_epochs", type=float, default=10)

# args = parser.parse_args()

# config = dict(
#     total_timesteps=args.total_timesteps,
#     policy='MlpPolicy',
#     n_steps_max=1000,
#     device='cpu',
#     clip_range=args.clip_range,
#     save_freq=10000,
#     learning_rate=args.learning_rate,
#     batch_size=int(args.n_steps*args.batch_size_multiple),
#     n_steps=args.n_steps,
#     n_epochs=args.n_epochs
# )

# run = wandb.init(project='OT2_RL_test',
#                  config=config,
#                  sync_tensorboard=True)

# env = WrappedEnv(render=False,
#                  max_step=config['n_steps_max'])

# model = PPO(policy='MlpPolicy',
#             env=env,
#             device=config['device'],
#             learning_rate=config['learning_rate'],
#             batch_size=config['batch_size'],
#             n_steps=config['n_steps'],
#             verbose=1)

# cb = WandbCallback()


    
    
# model.learn(total_timesteps=config['total_timesteps'],
#             callback=[cb, ClearMLCallback()])



# # for i in range(config['total_timesteps']//config['save_freq']):
# #     model.learn(total_timesteps=config['save_freq'],
# #                 callback=[cb, ClearMLCallback()],
# #                 reset_num_timesteps=False)
# #     model.save(f"models/RL_test/run_{run.id}/step_{config['save_freq']*(i+1)}")
    
# #     clearml.Logger.current_logger().report_scalar(
# #         title='Training Progress',
# #         series='Iteration',
# #         value=i + 1,
# #         iteration=config['save_freq'] * (i + 1)
#     # )
#     # wandb.log({
#     #     'Iteration': i + 1,
#     #     'Timesteps': config['save_freq'] * (i + 1)
#     # })
#     # clearml.Logger.current_logger().report_scalar(
#     #     title='Training Progress',
#     #     series='Iteration',
#     #     value=i + 1,
#     #     iteration=config['save_freq'] * (i + 1)
#     # )
# run.finish()










# import argparse
# import os
# import wandb
# from ot2_env_wrapper import WrappedEnv
# from stable_baselines3 import PPO
# from wandb.integration.sb3 import WandbCallback
# from clearml import Task


# # Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# # It can also be helpful to include the hyperparameters in the task name

# # task = Task.init(project_name='Mentor Group S/Group 1', task_name='Experiment1')
# task = Task.init(project_name='Mentor Group S/Bartosz Kudyba', task_name='Experiment-test')
# # copy these lines exactly as they are
# # setting the base docker image

# # copy these lines exactly as they are
# # setting the base docker image
# task.set_base_docker('deanis/2023y2b-rl:latest')
# # setting the task to run remotely on the default queue
# task.execute_remotely(queue_name="default")

# os.environ['WANDB_API_KEY'] = '47f4fed852265faa4acb02db524138518171f840'
# print(1)
# parser = argparse.ArgumentParser()
# parser.add_argument("--learning_rate", type=float, default=0.0003)
# parser.add_argument("--batch_size_multiple", type=float, default=0.05)
# parser.add_argument("--n_steps", type=int, default=1000)
# parser.add_argument("--n_epochs", type=int, default=10)
# parser.add_argument("--clip_range", type=float, default=0.2)
# parser.add_argument("--total_timesteps", type=int, default=1000000)
# args = parser.parse_args()
# print(2)
# config = dict(
#     total_timesteps=args.total_timesteps,
#     policy='MlpPolicy',
#     n_steps_max=1000,
#     device='cpu',
#     save_freq=10000,
#     learning_rate=args.learning_rate,
#     batch_size=int(args.n_steps*args.batch_size_multiple),
#     n_steps=args.n_steps,
#     n_epochs=args.n_epochs,
#     clip_range=args.clip_range
# )
# print(3)
# run = wandb.init(project='OT2_RL_test',
#                  config=config,
#                  sync_tensorboard=True)
# print(4)
# env = WrappedEnv(render=False,
#                  max_step=config['n_steps_max'])
# print(5)
# model = PPO(policy='MlpPolicy',
#             env=env,
#             device=config['device'],
#             tensorboard_log=f'runs/Rl_test_{run.id}',
#             learning_rate=args.learning_rate,
#             batch_size=int(args.n_steps*args.batch_size_multiple),
#             n_steps=args.n_steps,
#             n_epochs=args.n_epochs,
#             clip_range=args.clip_range)

# cb = WandbCallback(model_save_freq=config['save_freq'],
#                    model_save_path=f'models/RL_test/model_{run.id}')

# print(6)
# for i in range(config['total_timesteps']//config['save_freq']):
#     model.learn(total_timesteps=config['save_freq'],
#                 progress_bar=True,
#                 callback=cb,
#                 tb_log_name=f'runs/Rl_test_{run.id}',
#                 reset_num_timesteps=False)
#     model.save(f"models/RL_test/run_{run.id}/step_{config['save_freq']*(i+1)}")
#     print(7)



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
task = Task.init(project_name='Mentor Group S/Bartosz Kudyba', task_name='Experiment-RL')
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
parser.add_argument("--clip_range", type=float, default=0.2)
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
    n_epochs=args.n_epochs,
    clip_range=args.clip_range
)

run = wandb.init(project='OT2_RL',
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
            n_epochs=args.n_epochs,
            clip_range=config['clip_range']
            )

cb = WandbCallback(model_save_freq=config['save_freq'],
                   model_save_path=f'models/RL_test/model_{run.id}')


for i in range(config['total_timesteps']//config['save_freq']):
    model.learn(total_timesteps=config['save_freq'],
                progress_bar=True,
                callback=cb,
                tb_log_name=f'runs/Rl_test_{run.id}',
                reset_num_timesteps=False)
    model.save(f"models/RL_test/run_{run.id}/step_{config['save_freq']*(i+1)}")