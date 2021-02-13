from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
import pandas as pd
import os

derived_data_folder = "derived_data" + os.path.sep
datasets = os.listdir(derived_data_folder)
sequence_lengths = [3, 7]
lstm_log_csv_file = "lstm.log.csv"
learning_rate = 0.001
epochs = 10

if not os.path.exists(lstm_log_csv_file):
    log_file = open(lstm_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    for sequence_length in sequence_lengths:
        X = pd.read_csv(derived_data_folder + dataset)
        X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        y = X.pop('roch_rains_tomorrow')
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            X_train,
            y_train,
            sequence_length=sequence_length,
            sampling_rate=1,
            batch_size=256,
        )
        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            X_test,
            y_test,
            sequence_length=sequence_length,
            sampling_rate=1,
            batch_size=256,
        )

        for batch in dataset_train.take(1):
            inputs, targets = batch

        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        lstm_out = keras.layers.LSTM(32)(inputs)
        outputs = keras.layers.Dense(1)(lstm_out)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mse", metrics=['accuracy'])
        path_checkpoint = "checkpoints" + os.path.sep + str(dataset) + "." + str(sequence_length) + "model_checkpoint.h5"
        es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

        modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )

        history = model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[es_callback, modelckpt_callback],
        )

        acc = history.history["val_accuracy"][-1]
        print(dataset + "," + str(sequence_length) + "," + str(acc))
        with open(lstm_log_csv_file, 'a') as log_file:
            log_file.write(dataset + "," + str(sequence_length) + "," + str(acc) + "\n")
            log_file.close()
