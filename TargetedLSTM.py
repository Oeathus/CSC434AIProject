from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
import pandas as pd
import numpy as np
import os

from urllib.request import urlopen
import re
import os
import pandas as pd
from io import StringIO

dateFixer = re.compile(r'(\d{4}-\d{2}-\d{2})\s\d{2}:\d{2}')
spaceRemover = re.compile(r'(?<=,)\s{3}(?=,)')

roc_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" \
          "station=ROC&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&data=p01i&data=vsby" \
          "&year1=2020&month1=12&day1=9&year2=2020&month2=12&day2=19&tz=America%2FNew_York&format=onlycomma" \
          "&latlon=no&elev=no&missing=empty&trace=empty&direct=no&report_type=1&report_type=2"
roc_data = urlopen(roc_url, timeout=300).read().decode("utf-8")

roc_data = dateFixer.sub(r'\1', roc_data)
roc_data = spaceRemover.sub("", roc_data)
roc_data = re.sub("valid", "day", roc_data)

roc_data = pd.read_csv(StringIO(roc_data))

roc_data.drop(roc_data[roc_data['vsby'] < 0].index, inplace=True)
roc_data = roc_data.groupby('day').agg({'p01i': 'sum', 'tmpf': 'mean',
                                        'dwpf': 'mean', 'relh': 'mean',
                                        'drct': 'mean', 'sknt': 'mean',
                                        'alti': 'mean', 'mslp': 'mean',
                                        'vsby': 'mean'})

roc_data["roch_rained_today"] = (roc_data["p01i"] > 2) * 1
roc_data["roch_rained_today"] = roc_data["roch_rained_today"].astype(np.int64)

remember_row = roc_data.copy()

roc_data.drop(roc_data.columns[roc_data.columns.str.contains('p01i', case=False)], axis=1, inplace=True)
roc_data.reset_index(drop=True, inplace=True)

roc_data["roch_rains_tomorrow"] = 0
for i in range(roc_data.shape[0] - 1):
    roc_data.loc[i, "roch_rains_tomorrow"] = roc_data.loc[i + 1, "roch_rained_today"]

roc_data = roc_data[:-1]
roc_data["roch_rains_tomorrow"] = roc_data["roch_rains_tomorrow"].astype(np.int64)
roc_data_target = roc_data.pop("roch_rains_tomorrow")

normalize_cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "mslp", "vsby"]
for col in normalize_cols:
    roc_data[col] = (roc_data[col] - roc_data[col].min()) / (roc_data[col].max() - roc_data[col].min())

learning_rate = 0.001
epochs = 10

X = pd.read_csv("derived_data/without_others_2class_2_minmax.csv")
X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
y = X.pop('roch_rains_tomorrow')
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    y_train,
    sequence_length=7,
    sampling_rate=1,
    batch_size=256,
)
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    X_test,
    y_test,
    sequence_length=7,
    sampling_rate=1,
    batch_size=256,
)
dataset_predict = keras.preprocessing.timeseries_dataset_from_array(
    roc_data,
    roc_data_target,
    sequence_length=7,
    sampling_rate=1,
    batch_size=roc_data.shape[0]
)

for batch in dataset_train.take(1):
    inputs, targets = batch

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=['accuracy'])
path_checkpoint = "checkpoints" + os.path.sep + "lstm.model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

if not os.path.exists(path_checkpoint):
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=[es_callback, modelckpt_callback],
    )
else:
    model.load_weights(path_checkpoint)

prediction = (model.predict(dataset_predict) > 0.17) * 1
print(remember_row)
print(prediction)
