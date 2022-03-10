import matplotlib.pyplot as plt
import pandas as pd

dataset = 'CIFAR100'
batch_size = 256
trials = 10
EPOCHS = 200
finish_parallel_by=50

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

def write_list_to_csv(dictionary, heading):
    dataframe = pd.DataFrame(dictionary)
    dataframe.to_csv(heading + '.csv')
