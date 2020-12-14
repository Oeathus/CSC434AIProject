import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def input_fn(features, labels, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.batch(batch_size)


derived_data_folder = "derived_data" + os.path.sep
datasets = os.listdir(derived_data_folder)
random_states = [0, 42]
dnn_log_csv_file = "dnn.log.csv"
wnn_log_csv_file = "wnn.log.csv"
dwnn_log_csv_file = "dwnn.log.csv"

if not os.path.exists(dnn_log_csv_file):
    log_file = open(dnn_log_csv_file, 'w')
    log_file.close()

if not os.path.exists(wnn_log_csv_file):
    log_file = open(wnn_log_csv_file, 'w')
    log_file.close()

if not os.path.exists(dwnn_log_csv_file):
    log_file = open(dwnn_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    for random_state in random_states:
        X = pd.read_csv(derived_data_folder + dataset)
        X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        y = X.pop('roch_rains_tomorrow')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)
        classes = len(set(y.values))

        feature_columns = []
        for key in X.keys():
            if "rain" in key:
                col = tf.feature_column.categorical_column_with_identity(key, len(set(X[key].values)))
                feature_columns.append(tf.feature_column.indicator_column(col))
            else:
                feature_columns.append(tf.feature_column.numeric_column(key=key))

        Deep3 = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[54, 27, 6],
            n_classes=classes)
        Deep2 = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[27, 6],
            n_classes=classes)
        Wide = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            n_classes=classes)
        DeepWide3 = tf.estimator.DNNLinearCombinedClassifier(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=[54, 27, 6],
            n_classes=classes)
        DeepWide2 = tf.estimator.DNNLinearCombinedClassifier(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=[27, 6],
            n_classes=classes)

        Deep3.train(input_fn=lambda: input_fn(X_train, y_train))
        Deep2.train(input_fn=lambda: input_fn(X_train, y_train))
        Wide.train(input_fn=lambda: input_fn(X_train, y_train))
        DeepWide3.train(input_fn=lambda: input_fn(X_train, y_train))
        DeepWide2.train(input_fn=lambda: input_fn(X_train, y_train))

        Deep3_eval_result = Deep3.evaluate(input_fn=lambda: input_fn(X_test, y_test))
        Deep2_eval_result = Deep2.evaluate(input_fn=lambda: input_fn(X_test, y_test))
        Wide_eval_result = Wide.evaluate(input_fn=lambda: input_fn(X_test, y_test))
        DeepWide3_eval_result = DeepWide3.evaluate(input_fn=lambda: input_fn(X_test, y_test))
        DeepWide2_eval_result = DeepWide2.evaluate(input_fn=lambda: input_fn(X_test, y_test))
        print("For " + dataset + " with Random State " + str(random_state) + "\n")
        print(str(Deep3_eval_result["accuracy"]) + "\n")
        print(str(Deep2_eval_result["accuracy"]) + "\n")
        print(str(Wide_eval_result["accuracy"]) + "\n")
        print(str(DeepWide3_eval_result["accuracy"]) + "\n")
        print(str(DeepWide2_eval_result["accuracy"]) + "\n\n")

        with open(dnn_log_csv_file, 'a') as log_file:
            log_file.write(dataset + "," + str(random_state) + ","
                           + str(3) + "," + str(Deep3_eval_result["accuracy"]) + "\n")
            log_file.write(dataset + "," + str(random_state) + ","
                           + str(2) + "," + str(Deep2_eval_result["accuracy"]) + "\n")
            log_file.close()
        with open(wnn_log_csv_file, 'a') as log_file:
            log_file.write(dataset + "," + str(random_state) + "," + str(Wide_eval_result["accuracy"]) + "\n")
            log_file.close()
        with open(dwnn_log_csv_file, 'a') as log_file:
            log_file.write(dataset + "," + str(random_state) + ","
                           + str(3) + "," + str(DeepWide3_eval_result["accuracy"]) + "\n")
            log_file.write(dataset + "," + str(random_state) + ","
                           + str(2) + "," + str(DeepWide2_eval_result["accuracy"]) + "\n")
            log_file.close()
