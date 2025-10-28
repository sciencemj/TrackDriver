# Track Driver with Stable Baselines3
This project train PPO model to complete a given track using __Stable Baseline3__ and __Gymnasium__
## Train
```bash
$ python3 train_model.py
```
trained model saved on __models__ folder
and Tensorboard log saved on __logs__ folder

you can use tracks in __tracks__ folder to train
you can create tracks using __trackmaker.py__
```bash
$ tensorboard --logdir=./logs
```
to view logs
## Load Model
```bash
$ python3 load_model.py
```