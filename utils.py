import argparse
from math import log
import matplotlib.pyplot as plt
import numpy as np
from os import path, mkdir
import pandas as pd
import pickle
import sys


def on_press(event, var, val_var, fig, ax):
    if event.button != 1:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    val = int(x)
    if val < len(var) and val < len(val_var):
        minimum = min(min(var[val:]), min(val_var[val:]))
        maximum = max(max(var[val:]), max(val_var[val:]))
        ax[1].set_xlim(val, len(val_var))
        ax[1].set_ylim(minimum - (minimum * 0.1), maximum + (maximum * 0.1))
        fig.canvas.draw()


def display(car, val_car, title, param1, param2):
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].set(title=title, ylabel="Error", xlabel="Epochs")
    ax[1].set(title="Zoomed window")
    ax[0].plot(car, label=param1)
    ax[1].plot(range(0, len(car)), car[0:], label=param1)
    ax[0].plot(val_car, label=param2)
    ax[1].plot(range(0, len(val_car)), val_car[0:], label=param2)
    ax[0].legend(title="Parameter where:")
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: on_press(event, car, val_car, fig, ax),
    )
    return fig, ax


def check_hidden_layer(hl_list):
    try:
        hl_list = hl_list.split(",")
        result = tuple(int(i) for i in hl_list)
    except:
        sys.exit("Failed to create hidden layers")
    for i in result:
        if i < 1 or i > 9999:
            sys.exit("Failed to create hidden layers")
    return result


def open_datafile(datafile):
    try:
        # cols = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_ste", "texture_ste", "perimeter_ste", "area_ste", "smoothness_ste", "compactness_ste", "concavity_ste", "concave points_ste", "symmetry_ste", "fractal_dimension_ste", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
        data = pd.read_csv(datafile, header=None)  # , names=cols)
    except pd.errors.EmptyDataError:
        sys.exit("Empty data file.")
    except pd.errors.ParserError:
        sys.exit("Error parsing file, needs to be a well formated csv.")
    except:
        sys.exit(f"File {datafile} corrupted or does not exist.")
    return data


def normalize(df):
    result = df.copy()
    for feature_name in df.columns[1:]:
        if feature_name != 0:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (
                max_value - min_value
            )
    return result


def processing(real, n):
    predicted = []
    actual = np.copy(real)
    for val in actual:
        if val[0] == "B":
            val[0] = 0
        if val[0] == "M":
            val[0] = 1
    for index in range(len(actual)):
        predicted_values = n.query(
            np.array(actual[index][1:], dtype=np.float64)
        )[::-1]
        predicted.append(predicted_values)
    return actual[:, 0], predicted


def roc(real, n):
    actual, predicted = processing(real, n)
    pred = np.zeros(len(predicted))
    for i in range(len(predicted)):
        pred[i] = predicted[i][1]
    actual = np.int64(np.array(actual))
    thresholds = np.linspace(1, 0, 1001)
    roc = np.zeros((1001, 2))
    for i in range(1001):
        t = thresholds[i]
        tp = np.logical_and(pred > t, actual == 1).sum()
        tn = np.logical_and(pred <= t, actual == 0).sum()
        fp = np.logical_and(pred > t, actual == 0).sum()
        fn = np.logical_and(pred <= t, actual == 1).sum()
        r_fp = fp / float(fp + tn)
        roc[i, 0] = r_fp
        r_tp = tp / float(tp + fn)
        roc[i, 1] = r_tp
    auc = 0.0
    for i in range(1000):
        auc += (roc[i + 1, 0] - roc[i, 0]) * (roc[i + 1, 1] + roc[i, 1])
    auc *= 0.5
    plt.figure(figsize=(6, 6))
    plt.plot(roc[:, 0], roc[:, 1], "k", label="ROC curve")
    plt.plot([0, 1], "r--", label="Random guess")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve, AUC = {auc:.4f}")
    plt.show()


def binary_cross_entropy(real, n):
    actual, predicted = processing(real, n)
    sum_, good = 0, 0
    for index in range(len(actual)):
        sum_ += -log(predicted[index][actual[index]])
        if actual[index] == np.argmax(predicted[index]):
            good += 1
    error = (1 / len(actual)) * sum_
    acc = good / len(actual)
    return error, acc


def append_losses(n, data, validation_data):
    loss, acc = binary_cross_entropy(data, n)
    n.loss.append(loss)
    n.acc.append(acc)
    val_loss, val_acc = binary_cross_entropy(validation_data, n)
    n.val_loss.append(val_loss)
    n.val_acc.append(val_acc)
    return loss, val_loss, acc, val_acc


def load_model(file):
    try:
        with open(file, "rb") as fp:
            return pickle.load(fp)
    except:
        sys.exit(f"Error can't load file : {file}")


def save_model(model, n):
    i = 0
    if not path.exists("models"):
        try:
            mkdir("models")
        except OSError:
            sys.exit("Creation of the directory %s failed" % path)
    if model is True:
        while path.exists("models/model_" + str(i) + ".p"):
            i += 1
        with open("models/model_" + str(i) + ".p", "wb") as fp:
            pickle.dump(n, fp)
            print(f"Model saved in file models/model_{str(i)}.p")
    else:
        with open("models/model.p", "wb") as fp:
            pickle.dump(n, fp)
            print("Model saved in file models/model.p")


def get_seed(n):
    try:
        n = int(n)
    except (ValueError, TypeError):
        if n is None:
            n = np.random.randint(1, 2147483647)
        else:
            sys.exit(f"Value '{n}'' is not a correct value, need to be an int")
    np.random.seed(int(n))
    print(f"Seed:\t\t{np.random.get_state()[1][0]:>12}\n")


def split_size(n):
    value = int(n)
    if value not in range(1, 100):
        raise argparse.ArgumentTypeError(
            f"{value} is out of range, choose in [1-99]"
        )
    return value
