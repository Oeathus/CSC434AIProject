from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os

derived_data_folder = "derived_data" + os.path.sep
datasets = os.listdir(derived_data_folder)
random_states = [0, 42]
knn_log_csv_file = "knn.log.csv"
knn_log_file = "knn.log"

if not os.path.exists(knn_log_file):
    log_file = open(knn_log_file, 'w')
    log_file.close()

if not os.path.exists(knn_log_csv_file):
    log_file = open(knn_log_csv_file, 'w')
    log_file.close()

for dataset in datasets:
    for random_state in random_states:
        strongest_neighbor = 0
        best_score = 0
        best_classification = ""

        X = pd.read_csv(derived_data_folder + dataset)
        X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        y = X.pop('roch_rains_tomorrow')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.3)

        for i in range(1, 101):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            score = knn.score(X_test, y_test)
            if score > best_score:
                best_score = score
                strongest_neighbor = i
                best_classification = classification_report(y_test, y_pred)
        log_text = dataset + "," + str(random_state) + "," + str(strongest_neighbor) + "\n" + best_classification + "\n"
        print(log_text)
        with open(knn_log_csv_file, 'a') as log_file:
            log_file.write(dataset + "," + str(random_state) + ","
                           + str(strongest_neighbor) + "," + str(best_score) + "\n")
            log_file.close()
        with open(knn_log_file, 'a') as log_file:
            log_file.write(log_text)
            log_file.close()
