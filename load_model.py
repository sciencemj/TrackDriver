from stable_baselines3 import PPO
from environment import RaceEnvironment
import glob
import os

if __name__ == "__main__":
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

    models_path = "models/"
    model_files = glob.glob(os.path.join(models_path, "*"))
    model_files.sort(reverse=True)
    print("----Available Models----")
    if len(model_files) > 10:
        model_files = model_files[:10]
    for i in range(len(model_files)):
        print(f"[{i+1}]: {os.path.basename(model_files[i])}")
    while True:
        selected = int(input("Select Model: "))
        if selected < len(model_files) + 1:
            model_path = model_files[selected-1]
            break
    model_episodes = [os.path.basename(f) for f in glob.glob(os.path.join(model_path, "*.zip"))]
    model_episodes = [int(x.strip('.zip')) for x in model_episodes]
    model_episodes.sort(reverse=True)
    #print(model_episodes)
    model = PPO.load(model_path + "/" + str(model_episodes[0]), env=env)
    vec_env = model.get_env()

    for episode in range(5):
        obs = vec_env.reset()
        done = False
        while not done:
            action = model.predict(obs)
            obs, reward, done, info = vec_env.step(action)
            env.render()
        vec_env.close()