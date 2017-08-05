import pickle
from hyperas.utils import eval_hyperopt_space
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

trials = pickle.load(open("trials.p", "rb")) # Trials changed
space = pickle.load(open("space.p", "rb"))

for trial in trials:
    vals = trial.get('misc').get('vals')
    print(eval_hyperopt_space(space,vals))
    print(trial.get('result').get('loss'))