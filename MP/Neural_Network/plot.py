import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def plot_accuracy(accs):
    plt.figure()
    plt.plot(accs)
    plt.title("Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()