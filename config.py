import matplotlib.pyplot as plt
import pandas as pd

dataset = 'CIFAR100'
batch_size = 1024
trials = 10
EPOCHS = 200

def plot(epoch_range, train, test, heading, extra = 1):
    epochs = range(1,epoch_range+1)
    plt.plot(epochs, train, 'g', label='Training ' + heading)
    plt.plot(epochs, test, 'b', label='Test ' + heading)
    plt.title('Training and Test ' + heading)
    plt.xlabel('Epochs')
    plt.ylabel(heading)
    plt.legend()
    plt.savefig(heading + str(extra) + '.png')
    plt.close()

def write_list_to_csv(epoch_range, heading, list1, list2, extra = 1):
    epochs = range(0,epoch_range)
    dictionary = {'epochs': epochs, 'Timings': list1, 'Accuracy': list2}
    dataframe = pd.DataFrame(dictionary)
    dataframe.to_csv(heading + str(extra) + '.csv')
