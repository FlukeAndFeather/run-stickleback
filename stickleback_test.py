from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from stickleback.stickleback import Stickleback
from stickleback.util import align_events
from stickleback.visualize import outcome_table
import sys

from pdb import set_trace

## Read args
datapath = sys.argv[1]
win_size = int(sys.argv[2])
n_folds = int(sys.argv[3])
n_trees = int(sys.argv[4])
n_cores = int(sys.argv[5])

## Create output directory
outdir = os.path.join(os.path.dirname(datapath), datetime.now().strftime("run-stickleback-%Y%m%d%H%M%S"))
os.mkdir(outdir)

## Load data
sensors, events = pd.read_pickle(datapath)
for deployid in events:
    t1 = sensors[deployid].index[int(win_size / 2)]
    t2 = sensors[deployid].index[int(-win_size / 2)]
    events[deployid] = events[deployid][events[deployid].to_series().between(t1, t2)]
events = align_events(events, sensors)
for d in sensors:
    sensors[d] = sensors[d].interpolate().fillna(method="backfill")

# ## TEST ONLY
# ## Keep only first four deployments and select just 2 hours
# keep = list(sensors.keys())[0:4]
# sensors2 = dict()
# events2 = dict()
# for k in keep:
#     sensors2[k] = sensors[k].iloc[0:(2*3600*10)]
#     events2[k] = events[k][events[k] < sensors2[k].index[-1]]
#     if len(events2[k]) == 0:
#         raise RuntimeError("no events in first 2 hours of {}".format(k))
# sensors = sensors2
# events = events2

## Apply mask based on depth threshold
max_depth = 10 # m
is_shallow = lambda depth : (depth.rolling(win_size, center=True, min_periods=1).max() <= max_depth).to_numpy()
depth_mask = {k: is_shallow(v["depth"]) for k, v in sensors.items()}
  
## Initialize Stickleback
cols = sensors[list(sensors)[0]].columns
cif = ColumnEnsembleClassifier(
    estimators=[("CIF_" + c, CanonicalIntervalForest(n_estimators=n_trees, n_jobs=n_cores), [i]) 
                for i, c in enumerate(cols)]
)
lgt = LogisticRegression(class_weight="balanced")
sb = Stickleback(
    local_clf=cif,
    global_clf=lgt,
    win_size=win_size,
    tol=pd.Timedelta("3s"),
    nth=10,
    n_folds=2
)

def split_dict(dict, keys):
    dict1 = {k: v for k, v in dict.items() if k in keys}
    dict2 = {k: v for k, v in dict.items() if k not in keys}
    return dict1, dict2

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
    fit_start = datetime.now()
    print("fitting (start at {})...".format(fit_start.strftime("%Y-%m-%d %H:%M:%S")))
    sb.fit(sensors_train, events_train, mask=depth_mask)
    fit_finish = datetime.now()
    print("fitting done at {}".format(fit_finish.strftime("%Y-%m-%d %H:%M:%S")))

    ## Predict on test data and assess
    pred_start = datetime.now()
    print("predicting (start at {})...".format(pred_start.strftime("%Y-%m-%d %H:%M:%S")))
    event_pred = sb.predict(sensors_test)
    pred_finish = datetime.now()
    print("predicting done at {}".format(pred_finish.strftime("%Y-%m-%d %H:%M:%S")))
    event_outcomes = sb.assess(event_pred, events_test)

    ## Save output
    modelfile = "_".join(list(sensors_test.keys())) + "_model.pkl"
    with open(os.path.join(outdir, modelfile), 'wb') as f:
        pkl.dump(event_pred, f)
    print("Model saved to {}".format(modelfile))
    resultsfile = "_".join(list(sensors_test.keys())) + "_results.pkl"
    with open(os.path.join(outdir, resultsfile), 'wb') as f:
        pkl.dump(event_pred, f)
    print("Results saved to {}".format(resultsfile))

    ## Print result
    print(outcome_table(event_outcomes))
