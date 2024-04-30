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

    def _createModel(self):
        # print(self.__data.getTensorData())
        # tf.print(self.__data.getTensorData())
        # tf.io.write_file('tensorData', self.__data.getTensorData().numpy)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(self.__data.getNormalizedData()[0])
        self.__model = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Reshape((1,1)),
            # tf.keras.layers.LSTM(128, return_sequence=True),
            # tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # tf.keras.layers.LSTM(16, return_sequences=True),
            # tf.keras.layers.LSTM(8, return_sequences=True),
            # tf.keras.layers.LSTM(4, return_sequences=True),
            normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1),
        ])
        self.__model.compile(
            # optimizer='sgd',#tf.keras.optimizers.Adam(learning_rate=0.001), 
            # loss=tf.keras.losses.MeanSquaredError(),#tf.keras.losses.Huber(),#tf.keras.losses.LogCosh(),#'mse', 
            # metrics=['mae', 'mse',]#['accuracy']
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        self.__model.build()
        # , input_shape=(self.__data.getTensorData().shape[0], self.__data.getTensorData().shape[1],)
        # Tensordata has shape of: (20365, 284)

    def _saveModel(self):
        """Saves the model to a file using the name provided in self.__modelName
        """
        try:
            #self.__model.summary()
            self.__model.save(self.__modelName)
        except:
            print(str(time.ctime())+' - Could not save '+self.__modelName)

    def loadData(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        self.__data.readDataFromFile(filename, format)

    def readModel(self):
        """Reads a model from file using the name provided in self.__modelName
        """
        try:
            self.__model = tf.keras.models.load_model(self.__modelName)
            #self.__model.summary()
        except:
            self._createModel()

    def train(self, epochs = 1):
        # print(tf.shape(self.__data.getTensorData()[0]))
        print(self.__data.getData().shape)
        self.__model.fit(self.__data.getData(), self.__data.getData().columns, epochs=epochs, verbose=2, validation_split=0.2)
        # self.__model.fit(self.__data.getTensorData()[0], self.__data.getTensorData()[1], epochs=epochs, verbose=2, validation_split=0.2)
        # self.__model.fit(tf.convert_to_tensor(list(range(10))).numpy(), epochs=epochs, verbose=2)
        self._saveModel()
        
    def evaluate(self):
        pass
# end Arbiter