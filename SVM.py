from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import os

derived_data_folder = "derived_data" + os.path.sep
datasets = os.listdir(derived_data_folder)
random_states = [0, 42]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_log_csv_file = "svm.log.csv"
svm_log_file = "svm.log"

if not os.path.exists(svm_log_file):
    log_file = open(svm_log_file, 'w')
    log_file.close()

if not os.path.exists(svm_log_csv_file):
    log_file = open(svm_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    for kernel in kernels:
        for random_state in random_states:
            X = pd.read_csv(derived_data_folder + dataset)
            X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
            y = X.pop('roch_rains_tomorrow')
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

            sv_classifier = SVC(kernel=kernel)
            sv_classifier.fit(X_train, y_train)
            y_pred = sv_classifier.predict(X_test)

            log_text = dataset + "," + str(kernel) + "," + str(random_state) + "\n" + classification_report(y_test,
                                                                                                            y_pred) + "\n"
            print(log_text)
            with open(svm_log_csv_file, 'a') as log_file:
                log_file.write(dataset + "," + str(kernel) + "," + str(random_state) + "," + str(
                    sv_classifier.score(X_test, y_test)) + "\n")
                log_file.close()
            with open(svm_log_file, 'a') as log_file:
                log_file.write(log_text)
                log_file.close()
