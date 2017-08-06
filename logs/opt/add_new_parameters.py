from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.base import Ctrl
import pickle
import numpy as np
trials = pickle.load(open("trials.p", "rb"))


## DO ALL THE WORK HERE

specs, results, miscs, tids = trials.specs, trials.results, trials.miscs, trials.tids
new_miscs = []

def trial_change_1(t):
    only_idx = np.unique([i[0] for i in list(t['idxs'].values()) if i != []])
    assert len(only_idx)==1, "check index"
    t['idxs']['intermediate_dim'] = [only_idx[0]] # = [] if not the conditional !
    t['vals']['intermediate_dim'] = [1] # according to the number of the choice !
    t['idxs']['kernel_size'] = [only_idx[0]] # = [] if not the conditional !
    t['vals']['kernel_size'] = [2]

def trial_change_2(t):
    only_idx = np.unique([i[0] for i in list(t['idxs'].values()) if i != []])
    assert len(only_idx)==1, "check index"
    t['idxs']['middle_layer'] = [only_idx[0]] # = [] if not the conditional !
    t['vals']['middle_layer'] = [1] # according to the number of the choice !
    t['idxs']['epsilon'] = [] # = [] if not the conditional !
    t['vals']['epsilon'] = []
    t['idxs']['gaussian_regul'] = [] # = [] if not the conditional !
    t['vals']['gaussian_regul'] = []
    t['idxs']['dropout'] = [only_idx[0]]
    t['vals']['dropout'] = [0.2]

def trial_change_3(t):
    only_idx = np.unique([i[0] for i in list(t['idxs'].values()) if i != []])
    assert len(only_idx) == 1, "check index"
    if(t['vals']['middle_layer']['gaussian'] == [0]):
        bool_int_32 = True
        t['vals']['epsilon'] =[0] # epsilon was at 0.1

        t['idxs']['correct_factor'] = [only_idx[0]]
        t['vals']['correct_factor'] = [1]  # correct_factor was false
    else:
        t['idxs']['correct_factor'] = []
        t['vals']['correct_factor'] = []
    bool_int_32 = False # At the beginning of the loop
    if(bool_int_32):
        t['vals']['intermediate_dim'] = [1] # intermediate was at 32


for t in trials.miscs:
    print("TEST")


with open('./trials.p', 'wb') as f:
    pickle.dump(trials, f)


