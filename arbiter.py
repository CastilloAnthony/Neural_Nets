import tensorflow as tf
from dataHandler import DataHandler
import time
import logging
from pathlib import Path

class Arbiter():
    def __init__(self):
        self._setLogger()
        self._configureDirectories()
        self.__model = None
        self.__modelName = 'Arbiter-Test'
        self.__data = DataHandler()
        
    def __del__(self):
        pass

    def _configureFilename(self, timeData):
        currTimeString = str(timeData[0])
        if len(str(timeData[1])) == 1:
            currTimeString += '0'+str(timeData[1])
        else:
            currTimeString += str(timeData[1])
        if len(str(timeData[2])) == 1:
            currTimeString += '0'+str(timeData[2])
        else:
            currTimeString += str(timeData[2])
        return currTimeString
    
    def _setLogger(self):
        if not Path('./logs').is_dir():
            Path('./logs').mkdir()
            print(time.ctime()+' - ./logs directory has been created.')
        self.__currTime = time.localtime()
        logging.basicConfig(filename='./logs/'+self._configureFilename(self.__currTime)+'.log', encoding='utf-8', level=logging.DEBUG)
        logging.info(time.ctime()+' - Initializing...')
        logging.info(time.ctime()+' - Saving log to runtime_'+self._configureFilename(self.__currTime)+'.log')
        print(time.ctime()+' - Saving log to runtime_'+self._configureFilename(self.__currTime)+'.log')
    
    def _configureDirectories(self):
        if not Path('./data').is_dir():
            Path('./data').mkdir()
            logging.info(time.ctime()+' - ./data directory has been created.')
        if not Path('./graphs').is_dir():
            Path('./graphs').mkdir()
            logging.info(time.ctime()+' - ./graphs directory has been created.')
        if not Path('./graphs/availability').is_dir():
            Path('./graphs/availability').mkdir()
            logging.info(time.ctime()+' - ./graphs/availability directory has been created.')

    def _loadModel(self):
        pass

    def _createModel(self):
        self.__model = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Reshape((1,1)),
            # tf.keras.layers.LSTM(128, return_sequence=True),
            # tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # tf.keras.layers.LSTM(16, return_sequences=True),
            # tf.keras.layers.LSTM(8, return_sequences=True),
            # tf.keras.layers.LSTM(4, return_sequences=True),
            tf.keras.layers.Dense(1),
        ])
        self.__model.compile(
            optimizer='sgd',#tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss=tf.keras.losses.MeanSquaredError(),#tf.keras.losses.Huber(),#tf.keras.losses.LogCosh(),#'mse', 
            metrics=['mae', 'mse',]#['accuracy']
        )
        self.__model.build(input_shape=(32, 1,))
        self._saveModel()
    
    def _saveModel(self):
        """Saves the model to a file using the name provided in self.__modelFilename
        """
        try:
            #self.__model.summary()
            self.__model.save(self.__modelFilename)
        except:
            print(str(time.ctime())+' - Could not save '+self.__modelFilename)

    def _train(self):
        pass

    def _evaluate(self):
        pass

    def _loadData(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        self.__data.readDataFromFile(filename, format)
# end Arbiter