### Developed by Anthony Castillo ###
from arbiter import Arbiter

def main():
    conv_width, predictions = 24, 24
    newArbiter = Arbiter()
    newArbiter.loadData()
    newArbiter.readModel(conv_width=conv_width, predictions=predictions)
    newArbiter.train()
    newArbiter.evaluate()
    history = []
    for i in range(0,200000):
        print('Attempt: '+str(i))
        # newArbiter.loadData(predictions=predictions)
        newArbiter.randomizeTarget()
        newArbiter.recreateWindow(conv_width=conv_width, predictions=predictions)
        history.append(newArbiter.train())
        # print('\nHistory:')
        # print(history[-1].params)
        # print(history[-1].history.keys())
        # for i in history[-1].history:
        #     print(i, history[-1].history[i])
        newArbiter.evaluate()
    # history = newArbiter.train()
    print('\nHistory:')
    print(history[-1].params)
    print(history[-1].history.keys())
    for i in history[-1].history:
        print(i, history[-1].history[i])
    newArbiter.evaluate()
if __name__ == '__main__':
    main()