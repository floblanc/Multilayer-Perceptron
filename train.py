from activations_fun import sigmoid
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from neuralNetwork import neuralNetwork
import numpy as np
from utils import (
    open_datafile,
    normalize,
    save_model,
    display,
    check_hidden_layer,
    get_seed,
    append_losses,
    split_size,
)


def fit(args, n):
    epochs = args.epochs
    df = args.dataset_train[
        [1, 2, 3, 8, 11, 12, 17, 18, 19, 21, 26, 28, 30, 31]
    ]
    size_train = int(len(df) * args.split / 100)
    if args.verbose is True:
        print(f"Total data:\t\t{len(df):>4} values")
        print(f"Train data:\t\t{size_train:>4} values, {args.split:>3}%")
        print(
            f"Val data:\t\t{len(df) - size_train:>4} values, {100 - args.split:>3}%\n"
        )
        print(f"Input:\t\t\t{n.input:>4} neurones")
        for layer in range(len(n.hidden)):
            print(f"Hidden_{layer + 1}:\t\t{n.hidden[layer]:>4} neurones")
        print(f"Output:\t\t\t{n.output:>4} neurones\n")
        print(f"Epochs:\t\t\t{epochs:>4}")
        print(f"Learning Rate:\t\t{n.lr:>4}")
        print(f"Patience:\t\t{args.patience:>4}\n\n")
    train = df.iloc[:size_train, :]
    test = df.iloc[size_train + 1 :, :]
    data = normalize(train)
    validation_data = normalize(test)
    data = np.array(data)
    validation_data = np.array(validation_data)
    best_val_loss = 10
    patience = 0
    loss, val_loss, acc, val_acc = append_losses(n, data, validation_data)
    print(
        f"epoch {0:>3}/{epochs:<3} - loss: {loss:10.10f} - acc {acc:5.5f} - val_loss: {val_loss:10.10f} - val_acc {val_acc:5.5f}",
        end="\r",
    )
    for e in range(epochs):
        # np.random.shuffle(data)
        for values in data:
            targets = np.zeros(output_n) + 0.01
            if values[0] == "M":
                targets[0] = 0.99
            elif values[0] == "B":
                targets[1] = 0.99
            n.train(np.array(values[1:], dtype=np.float64), targets)
        loss, val_loss, acc, val_acc = append_losses(n, data, validation_data)
        print(
            f"epoch {e + 1:>3}/{epochs:<3} - loss: {loss:10.10f} - acc {acc:5.5f} - val_loss: {val_loss:10.10f} - val_acc {val_acc:5.5f}",
            end="\r",
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved = deepcopy(n.w)
            patience = 0
        else:
            if patience == args.patience:
                epoch = e - args.patience - 1
                print(f"\nEarly stopping, step back to epoch {epoch}")
                n.w = saved[:]
                n.loss = n.loss[: epoch + 2]
                n.val_loss = n.val_loss[: epoch + 2]
                n.acc = n.acc[: epoch + 2]
                n.val_acc = n.val_acc[: epoch + 2]
                print(
                    f"loss: {n.loss[-1]:10.10f} - acc {n.acc[-1]:5.5f} - val_loss: {n.val_loss[-1]:10.10f} - val_acc {n.val_acc[-1]:5.5f}"
                )
                break
            patience += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset_train", type=open_datafile)
    parser.add_argument("-b", "--bias", help="Enable Bias", action="store_true")
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="e",
        help="Choose number of epochs",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-vb",
        "--verbose",
        action="store_true",
        help="Enable verbose",
    )
    parser.add_argument(
        "-lr",
        "--learningrate",
        metavar="lr",
        help="Choose learning rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-p",
        "--patience",
        metavar="n",
        help="Choose patience for early stopping",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-hl",
        "--hidden_layer",
        metavar="(n1, n2, ...)",
        help="Make your own hidden layers",
        type=check_hidden_layer,
        default=(100, 50),
    )
    parser.add_argument(
        "-vi", "--visu", help="Display graphs", action="store_true"
    )
    parser.add_argument(
        "-s", "--seed", metavar="n", help="Choose seed", default=None
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Save model in different file",
        action="store_true",
    )
    parser.add_argument(
        "--split",
        metavar="[1-99]",
        help="Choose size of split",
        choices=(range(1, 100)),
        type=split_size,
        default=80,
    )

    args = parser.parse_args()
    get_seed(args.seed)
    input_n = 13
    output_n = 2
    hidden_layers = args.hidden_layer

    n = neuralNetwork(
        input_n, output_n, hidden_layers, args.learningrate, sigmoid, args.bias
    )
    fit(args, n)
    print()
    if args.visu is True:
        fig1, ax1 = display(
            n.loss, n.val_loss, "Loss Trend", "loss", "val_loss"
        )
        fig2, ax2 = display(
            n.acc, n.val_acc, "Accuracy Trend", "acc", "val_acc"
        )
        plt.show()
    save_model(args.model, n)
