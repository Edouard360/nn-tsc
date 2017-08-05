from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.base import Ctrl
import pickle

trials = pickle.load(open("trials.p", "rb"))


## DO ALL THE WORK HERE

specs, results, miscs, tids = trials.specs, trials.results, trials.miscs, trials.tids
new_miscs = []
for t in trials.miscs:
    # t['idxs']['new field'] = [] # new field that you can put to None, if on another branch and not used
    # t['vals']['known_field'] = [0] # if you forgot to mention one field but still want to put it
    # t['idxs'].pop('field_to_suppress', None)
    new_miscs += [t]

# trials.idxs_vals # VERY IMPORTANT !!
# trials.idxs
# trials.vals # VERY IMPORTANT
# trials.miscs -> contains as key idxs ! and vals ! Probably doesn't need to change 'idxs' but yes for 'vals'

##


new_trials = Trials()
ctrl = Ctrl(new_trials)
ctrl.inject_results(specs, results, new_miscs, tids)
trials = ctrl.trials

with open('logs/opt/trials_changed.p', 'wb') as test:
    pickle.dump(trials, test)


