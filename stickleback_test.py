import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from stickleback.stickleback import Stickleback
from stickleback.util import split_dict
from stickleback.visualize import outcome_table
import sys

## Read args
datapath = sys.argv[1]
win_size = int(sys.argv[2])
n_trees = int(sys.argv[3])

## Load data
sensors, events = pd.read_pickle(datapath)
for deployid in events:
    t1 = sensors[deployid].index[int(win_size / 2)]
    t2 = sensors[deployid].index[int(-win_size / 2)]
    events[deployid] = pd.DatetimeIndex(events[deployid][events[deployid].between(t1, t2)])
  
## Initialize Stickleback
cols = sensors[list(sensors)[0]].columns
tsf = ColumnEnsembleClassifier(
    estimators=[("TSF_" + c, TimeSeriesForestClassifier(n_estimators=n_trees, n_jobs=-1), [i]) 
                for i, c in enumerate(cols)]
)
knn = KNeighborsClassifier(3)
sb = Stickleback(
    local_clf=tsf,
    global_clf=knn,
    win_size=win_size,
    tol=pd.Timedelta("5s"),
    nth=5
)

## Split test-train data
test_ids = list(sensors.keys())[0:2]
sensors_test, sensors_train = split_dict(sensors, test_ids)
events_test, events_train = split_dict(events, test_ids)

## Fit to training data
sb.fit(sensors_train, events_train)

## Predict on test data and assess
event_pred = sb.predict(sensors_test)
event_outcomes = sb.assess(event_pred, events_test)

## Print result
print(outcome_table)
