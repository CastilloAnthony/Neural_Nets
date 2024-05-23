### Developed by Anthony Castillo ###
from arbiter import Arbiter

def main():
    history = {}
    conv_width, predictions = 24, 24
    newArbiter = Arbiter()
    newArbiter.loadData() # Data must always be loaded before anything else happens. A filename can be specified to target another aeronet file.
    newArbiter.readModel(conv_width=conv_width, predictions=predictions)
    history[newArbiter.getTarget()] = newArbiter.train()
    newArbiter.evaluate()
    for i in range(0,10000):
        print('Attempt: '+str(i))
        # newArbiter.randomizeTarget() # Randomizes the target amongst the valid wavelengths
        newArbiter.shiftTarget() # Increments the target to the next valid wavelength
        newArbiter.readModel(conv_width=conv_width, predictions=predictions)
        history[newArbiter.getTarget()] = newArbiter.train()
        newArbiter.evaluate()
    print('\nHistory:')
    for i in history:
        print('\n'+i)
        print(history[i].params)
        print(history[i].history.keys())
        for k in history[i].history:
            print(k, history[i].history[k])
# end main

if __name__ == '__main__':
    main()