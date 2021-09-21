import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sktime.transformations.panel.rocket import Rocket
from stickleback.stickleback import Stickleback
from stickleback.util import align_events, split_dict
from stickleback.visualize import outcome_table
import sys

## Read args
datapath = sys.argv[1]
win_size = int(sys.argv[2])
n_folds = int(sys.argv[3])
n_trees = int(sys.argv[4])

## Load data
sensors, events = pd.read_pickle(datapath)
for deployid in events:
    t1 = sensors[deployid].index[int(win_size / 2)]
    t2 = sensors[deployid].index[int(-win_size / 2)]
    events[deployid] = events[deployid][events[deployid].to_series().between(t1, t2)]
events = align_events(events, sensors)
for d in sensors:
    sensors[d] = sensors[d].interpolate().fillna(method="backfill")
  
## Initialize Stickleback
rkt = make_pipeline(
    Rocket(),
    LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000)
)
lgt = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000)
sb = Stickleback(
    local_clf=rkt,
    global_clf=lgt,
    win_size=win_size,
    tol=pd.Timedelta("5s"),
    nth=5
)

## k-fold validation
print("beginning {}-fold validation".format(n_folds))
kf = KFold(n_folds)
deployids = np.array(list(sensors.keys()))
for train_idx, test_idx in kf.split(deployids):
    test_ids = deployids[test_idx]
    sensors_test, sensors_train = split_dict(sensors, test_ids)
    events_test, events_train = split_dict(events, test_ids)
    print("training on {}".format(list(sensors_train.keys())))
    print("testing on {}".format(list(sensors_test.keys())))

    ## Fit to training data
    print("fitting...")
    sb.fit(sensors_train, events_train)
    print("done")

    ## Predict on test data and assess
    print("predicting...")
    event_pred = sb.predict(sensors_test)
    print("done")
    event_outcomes = sb.assess(event_pred, events_test)

    ## Print result
    print(outcome_table(event_outcomes))
