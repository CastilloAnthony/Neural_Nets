### Developed by Anthony Castillo ###
import time
import logging
from pathlib import Path
import tensorflow as tf
from dataHandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np
from random import randint
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# from windowGenerator import WindowGenerator
# https://www.tensorflow.org/install/pip
# https://www.tensorflow.org/guide/gpu

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

    def _createModel(self, predictions=6):
        """Creates a new multi step dense model
        """
        print('Number of Features: ', self.__data.getNumFeatures())
        self.__model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Flatten(),
            
            # tf.keras.layers.Dense(units=1024, activation='relu'),
            # tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=self.__data.getNumFeatures(), activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=1),

            # Single Output
            # tf.keras.layers.Dense(units=64, activation='relu'),
            # tf.keras.layers.Dense(units=64, activation='relu'),
            # tf.keras.layers.Dense(units=1),

            # Multi-Output
            # tf.keras.layers.Dense(units=64, activation='relu'),
            # tf.keras.layers.Dense(units=64, activation='relu'),
            # tf.keras.layers.Dense(units=self.__data.getNumFeatures()),
            # tf.keras.layers.Dense(units=len(self.__data.getValidWavelengths())),
            # tf.keras.layers.Dense(units=predictions),

            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
            
            # Shape => [batch, out_steps*features]
            # tf.keras.layers.Dense(predictions*len(self.__data.getValidWavelengths()), kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            # tf.keras.layers.Reshape([predictions, len(self.__data.getValidWavelengths())]),
            # tf.keras.layers.Reshape([len(self.__data.getValidWavelengths())]),
        ])

    def _saveModel(self):
        """Saves the model to a file using the name provided in self.__modelName
        """
        try:
            #self.__model.summary()
            self.__model.save('models/'+self.__modelName+'_'+str(time.time())+'.keras')
            print(str(time.ctime())+' - Model Saved as '+self.__modelName+'_'+str(time.time())+'.keras')
        except:
            print(str(time.ctime())+' - Could not save '+self.__modelName)

    def readModel(self, predictions=6):
        """Reads a model from file using the name provided in self.__modelName
        """
        try:
            self.__model = tf.keras.models.load_model(self.__modelName+'_checkpoint.keras')
            #self.__model.summary()
        except:
            self._createModel(predictions)

    def loadData(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv', target='AOD_380nm-Total', conv_width=24, predictions=6):
        self.__data.readDataFromFile(filename, format)
        self.__data.setTarget()
        self.__data.createWindow(conv_width=conv_width, predictions=predictions)

    def randomizeTarget(self):
        choice = randint(0, len(self.__data.getValidWavelengths())-1)
        target = self.__data.getValidWavelengths()[choice]
        self.__data.setTarget(target)
        print('Now training on '+target)

    def recreateWindow(self):
        self.__data._createWindow()

    def train(self, maxEpochs = 2000, totalPatience=2*50):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=totalPatience,
            mode='min',
            )

        auto_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.__modelName+'_checkpoint.keras',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            )
        
        self.__model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

        history = self.__model.fit(self.__data.getWindowTrainData(), 
                                   epochs=maxEpochs, 
                                   validation_data=self.__data.getWindowTrainValidation(),
                                   callbacks=[early_stopping, auto_save],# 
                                   )

        self._saveModel()
        # self.evaluate()
        # self.__data.plotModel(model=None)
        
        return history
        
    def evaluate(self):
        # plt.show()
        val_performance, performance = {}, {}
        val_performance['Model'] = self.__model.evaluate(self.__data.getWindowTrainValidation(), return_dict=True)
        performance['Model'] = self.__model.evaluate(self.__data.getWindowTrainTest(), verbose=0, return_dict=True)
        x = np.arange(len(performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        val_mae = [v[metric_name] for v in val_performance.values()]
        test_mae = [v[metric_name] for v in performance.values()]
        plt.figure(figsize=(12, 8))
        plt.ylabel(f'mean_absolute_error [{self.__data.getTarget()}, normalized]')
        plt.bar(x - 0.17, val_mae, width, label='Validation Set')
        plt.bar(x + 0.17, test_mae, width, label='Test Set')
        plt.gcf().suptitle('Model Performance')
        plt.xticks(ticks=x, labels=performance.keys(),
                rotation=45)
        _ = plt.legend()
        # plt.show()
        self.__data.plotModel(self.__model)
# end Arbiter