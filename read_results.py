#!/usr/bin/python
import pickle
import numpy as np

with open('results.pkl', 'rb') as f:
    results_train_time_for_paper = np.zeros((75,))
    results_test_accuracy_for_paper = np.zeros((75,))
    results = pickle.load(f)
    for index, model in enumerate(results):
        results_train_time_for_paper[index] = np.around(results[model]['train_time']/100, 1)
        results_test_accuracy_for_paper[index] = np.around(results[model]['test_accuracy'], 1)
    results_train_time_for_paper = results_train_time_for_paper.reshape(5, 15)
    results_test_accuracy_for_paper = results_test_accuracy_for_paper.reshape(5, 15)
    print(" \\\\\n".join([" & ".join(map(str, line)) for line in results_train_time_for_paper]))
    print(" \\\\\n".join([" & ".join(map(str, line)) for line in results_test_accuracy_for_paper]))
