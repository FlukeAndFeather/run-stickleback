from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle as pkl
import sktime.classification as tsc
import stickleback.stickleback as sb
import stickleback.util as sb_util
import stickleback.visualize as sb_viz
import sys

from pdb import set_trace

## Read args
datapath = sys.argv[1]
win_size = int(sys.argv[2])
n_fit = int(sys.argv[3])
n_kerns = int(sys.argv[4])
n_est = int(sys.argv[5])
n_min = float(sys.argv[6])
n_cores = int(sys.argv[7])

## Create output directory
outdir = os.path.join(os.path.dirname(datapath), datetime.now().strftime("run-stickleback-%Y%m%d%H%M%S"))
os.mkdir(outdir)

## Load data
sensors, events = pd.read_pickle(datapath)

## TEST ONLY
## Subset deployments and keep only the first few hours of each
keep = list(sensors.keys())[0:12]
sensors2 = dict()
events2 = dict()
max_hours = 4
for k in keep:
    sensors2[k] = sensors[k].iloc[0:(max_hours * 3600 * 10)]
    events2[k] = events[k][events[k] < sensors2[k].index[-1]]
    if len(events2[k]) == 0:
        raise RuntimeError("no events in first {} hours of {}".format(max_hours, k))
sensors = sensors2
events = events2

# Remove events near boundaries
for deployid in events:
    t1 = sensors[deployid].index[int(win_size / 2)]
    t2 = sensors[deployid].index[int(-win_size / 2)]
    events[deployid] = events[deployid][events[deployid].to_series().between(t1, t2)]
events = sb_util.align_events(events, sensors)
for d in sensors:
    sensors[d] = sensors[d].interpolate().fillna(method="backfill")

## Apply mask based on depth threshold
max_depth = 20 # m
is_shallow = lambda depth : (depth.rolling(win_size, center=True, min_periods=1).max() <= max_depth).to_numpy()
depth_mask = {k: is_shallow(v["depth"]) for k, v in sensors.items()}
  
## Initialize Stickleback
cif = tsc.interval_based.CanonicalIntervalForest(n_estimators = n_est,
                                                 base_estimator = "CIT",
                                                 n_jobs = n_cores)
ars = tsc.kernel_based.Arsenal(num_kernels=n_kerns, 
                               n_estimators=n_est, 
                               time_limit_in_minutes=n_min, 
                               n_jobs=n_cores)
# cols = sensors[list(sensors)[0]].columns
# ars = ColumnEnsembleClassifier(
#     estimators=[("ARS_" + c, Arsenal(num_kernels=n_kerns, n_estimators=n_est, time_limit_in_minutes=60.0, n_jobs=n_cores), [i]) 
#                 for i, c in enumerate(cols)]
# )
# lgt = LogisticRegression(class_weight="balanced")
sb = Stickleback(
    local_clf=cif,
    win_size=win_size,
    tol=pd.Timedelta("3s"),
    nth=5,
    n_folds=4,
    max_events = 1000
)

def split_dict(dict, keys):
    dict1 = {k: v for k, v in dict.items() if k in keys}
    dict2 = {k: v for k, v in dict.items() if k not in keys}
    return dict1, dict2

## fit, predict, save
deployids = np.array(list(sensors.keys()))
train_ids = deployids[0:n_fit]
sensors_train, sensors_test = split_dict(sensors, train_ids)
events_train, events_test = split_dict(events, train_ids)
print("training on {}".format(list(sensors_train.keys())))
print("testing on {}".format(list(sensors_test.keys())))

## Fit to training data
fit_start = datetime.now()
print("fitting (start at {})...".format(fit_start.strftime("%Y-%m-%d %H:%M:%S")))
sb.fit(sensors_train, events_train) #, mask=depth_mask)
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
outfile = "_".join(list(sensors_test.keys())) + ".pkl"
sb_util.save_fitted(sb, outfile, sensors, events, None, event_pred)
print("Results saved to {}".format(outfile))

## Print result
print(sb_viz.outcome_table(event_outcomes))
