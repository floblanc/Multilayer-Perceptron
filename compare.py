import argparse
import matplotlib.pyplot as plt
from utils import load_model
import sys
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def nearest_square(val):
    answer = 0
    while answer ** 2 < val:
        answer += 1
    return answer


def display_all(args, value):
    n = nearest_square(len(args))
    fig = plt.figure(figsize=(n * 3, n * 2), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=n, nrows=n, figure=fig)
    fig.suptitle(value, fontsize=16)
    f_ax = []
    i = 0
    for row in range(n):
        for col in range(n):
            if i < len(args):
                f_ax.append(fig.add_subplot(spec[row, col]))
                f_ax[-1].set(ylabel="Error", xlabel="Epochs")
                if value == "loss and val_loss":
                    f_ax[-1].plot(args[i].loss, label="loss")
                    f_ax[-1].plot(args[i].val_loss, label="val_loss")
                if value == "acc and val_acc":
                    f_ax[-1].plot(args[i].acc, label="acc")
                    f_ax[-1].plot(args[i].val_acc, label="val_acc")
                f_ax[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
            i += 1
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "models",
        nargs="*",
        type=load_model,
        help="models to select for comparison (16 maximum)",
    )
    args = parser.parse_args()
    if len(args.models) > 16:
        sys.exit(f"Too much models to plot : {len(args.models)} models")
    display_all(args.models, "loss and val_loss")
    display_all(args.models, "acc and val_acc")
