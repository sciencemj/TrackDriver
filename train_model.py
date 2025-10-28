#! /usr/bin/env python3
from environment import RaceEnvironment
from stable_baselines3 import PPO
import time
import os
import glob

EPISODES = 10

if __name__ == "__main__":
    models_dir = "models/PPO" + str(int(time.time()))
    log_dir = "logs/PPO" + str(int(time.time()))

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tracks_path = "tracks/"
    track_files = glob.glob(os.path.join(tracks_path, "*.track"))
    print("----Available Tracks----")
    for i in range(len(track_files)):
        print(f"[{i+1}]: {os.path.basename(track_files[i])}")
    while True:
        selected = int(input("Select Track: "))
        if selected < len(track_files) + 1:
            track_path = track_files[selected-1]
            break

    angle = int(input("Start Angle: "))

    env = RaceEnvironment(track_path, start_angle=angle)
    env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    for episode in range(1, EPISODES + 1):
        model.learn(total_timesteps=5000, reset_num_timesteps=False)
        model.save(f"{models_dir}/{episode}")

