### Developed by Anthony Castillo ###
from arbiter import Arbiter
from matplotlib import pyplot as plt
import numpy as np
import time

def main():
    history = {}
    conv_width, predictions = 24, 24
    newArbiter = Arbiter()
    newArbiter.loadData() # Data must always be loaded before anything else happens. A filename can be specified to target another aeronet file.
    newArbiter.readModel(conv_width=conv_width, predictions=predictions)
    history[newArbiter.getTarget()] = newArbiter.train()
    newArbiter.evaluate()
    for i in range(0, len(newArbiter.getValidWavelengths())-1):
        print('Attempt: '+str(i))
        # newArbiter.randomizeTarget() # Randomizes the target amongst the valid wavelengths
        newArbiter.shiftTarget() # Increments the target to the next valid wavelength
        newArbiter.readModel(conv_width=conv_width, predictions=predictions)
        history[newArbiter.getTarget()] = newArbiter.train()
        newArbiter.evaluate()
    # print('\nHistory:')
    # for i in history:
    #     print('\n'+i)
    #     print(history[i].params)
    #     print(history[i].history.keys())
    #     for k in history[i].history:
    #         print(k, history[i].history[k])
    plot_history(history)
# end main

def plot_history(history):
    history_latest = {}
    # loss, mean_absolute_error, val_loss, val_mean_absolute_error = {}, {}, {}, {}
    for i in history:
        # print('\n'+i)
        history_latest[i] = {'loss':0, 'mean_absolute_error':0, 'val_loss':0, 'val_mean_absolute_error':0}
        # print(history[i].params)
        # print(history[i].history.keys())
        for k in history[i].history:
            # print(k, history[i].history[k][-1])
            history_latest[i][k] = history[i].history[k][-1]
    print('\nHistory:')
    for i in history_latest:
        print(i, history_latest[i])

    # val_performance[self.__data.getTarget()] = self.__model.evaluate(self.__data.getWindowTrainValidation(), return_dict=True)
    # performance[self.__data.getTarget()] = self.__model.evaluate(self.__data.getWindowTrainTest(), verbose=0, return_dict=True)
    x = np.arange(len(history_latest))
    width = 0.1
    # print(x)
    plt.figure(figsize=(16, 9))
    plt.ylabel('Results')
    # labels = {}
    for i in x:
        plt.bar(i - width*2, history_latest[list(history_latest.keys())[i]]['loss'], width, color='C0')
        plt.bar(i - width*0.75, history_latest[list(history_latest.keys())[i]]['mean_absolute_error'], width, color='C1')
        plt.bar(i + width*0.75, history_latest[list(history_latest.keys())[i]]['val_loss'], width, color='C2')
        plt.bar(i + width*2, history_latest[list(history_latest.keys())[i]]['val_mean_absolute_error'], width, color='C3')
    
    plt.gcf().suptitle('Training History Results')
    plt.xticks(ticks=x, labels=history_latest.keys(), rotation=30)
    _ = plt.legend(labels=['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error'], labelcolor=['C0', 'C1', 'C2', 'C3',])
    plt.tight_layout()
    plt.savefig('graphs/'+'training_history.png')
    # plt.show()
    print(time.ctime()+' - New graphs generated in ./graph/ folder:\n\t'+'graphs/'+'training_history.png')
# end evaluate

if __name__ == '__main__':
    main()