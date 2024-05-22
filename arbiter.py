### Developed by Anthony Castillo ###
import time
import logging
from pathlib import Path
import tensorflow as tf
from dataHandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from baseline import Baseline
from residualWrapper import ResidualWrapper

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# from windowGenerator import WindowGenerator
# https://www.tensorflow.org/install/pip
# https://www.tensorflow.org/guide/gpu

class Arbiter():
    def __init__(self):
        self._configureDirectories()
        self.__model = None
        self.__modelName = 'Arbiter'
        self.__data = DataHandler()
        
    def __del__(self):
        del self.__model, self.__modelName, self.__data

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
        self.__currTime = time.localtime()
        logging.basicConfig(filename='./logs/'+self._configureFilename(self.__currTime)+'.log', encoding='utf-8', level=logging.DEBUG)
        logging.info(time.ctime()+' - Initializing...')
        logging.info(time.ctime()+' - Saving log to runtime_'+self._configureFilename(self.__currTime)+'.log')
        print(time.ctime()+' - Saving log to runtime_'+self._configureFilename(self.__currTime)+'.log')
    
    def _configureDirectories(self):
        if not Path('./logs').is_dir():
            Path('./logs').mkdir()
            print(time.ctime()+' - ./logs directory has been created.')
        self._setLogger()
        if not Path('./data').is_dir():
            Path('./data').mkdir()
            logging.info(time.ctime()+' - ./data directory has been created.')
        if not Path('./graphs').is_dir():
            Path('./graphs').mkdir()
            logging.info(time.ctime()+' - ./graphs directory has been created.')
        if not Path('./models').is_dir():
            Path('./models').mkdir()
            logging.info(time.ctime()+' - ./models directory has been created.')
        if not Path('./models/checkpoints').is_dir():
            Path('./models/checkpoints').mkdir()
            logging.info(time.ctime()+' - ./models/checkpoints directory has been created.')
        # if not Path('./graphs/availability').is_dir():
        #     Path('./graphs/availability').mkdir()
        #     logging.info(time.ctime()+' - ./graphs/availability directory has been created.')

    def _createModel(self):#, conv_width=24, predictions=24):
        """Creates a new residual long short-term memory model
        """
        # print(tf.keras.config.floatx())
        tf.keras.backend.set_floatx('float64')
        # print(tf.keras.config.floatx())
        self.__model = ResidualWrapper (
            tf.keras.Sequential([
            # Multi-Output Residual_lstm
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(self.__data.getNumFeatures(),
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros()
                )
        ]))

        self.__model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
            )
        logging.info(time.ctime()+' - New model created.')

    def _saveModel(self):
        """Saves the model to a file using the name provided in self.__modelName
        """
        try:
            #self.__model.summary()
            self.__model.save('models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            logging.info(time.ctime()+' - Model Saved as '+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            print(str(time.ctime())+' - Model Saved as '+self.__modelName+'_'+self.__data.getTarget()+'.keras')
        except:
            logging.info(time.ctime()+' - Could not save '+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            print(str(time.ctime())+' - Could not save '+self.__modelName+'_'+self.__data.getTarget()+'.keras')


    def readModel(self, conv_width=24, predictions=6):
        """Reads a model from file using the name provided in self.__modelName
        """
        self.__data.createWindow(conv_width=conv_width, predictions=predictions, label_columns=False)
        try:
            self.__model = tf.keras.models.load_model('models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            logging.info(time.ctime()+' - Found and loaded model '+'models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            print(time.ctime()+' - Found and loaded model '+'models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            # self.evaluate()
        except Exception as error1:
            try:
                logging.info(time.ctime()+' - '+str(error1))
                print(error1)
                self.__model = tf.keras.models.load_model('models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras')
                logging.info(time.ctime()+' - Found and loaded model '+'models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras')
                print(time.ctime()+' - Found and loaded model '+'models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras')
                # self.evaluate()
            except Exception as error2:
                logging.info(time.ctime()+' - '+str(error1))
                print(error2)
                logging.info(time.ctime()+' - Colud not find model for '+self.__data.getTarget())
                print(time.ctime()+' - Colud not find model for '+self.__data.getTarget())
                self._createModel()

    def loadData(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        self.__data.readDataFromFile(filename, format)
        self.__data.setTarget()
        logging.info(time.ctime()+' - Target set to '+self.__data.getTarget()+' and data loaded from '+filename)
        print(time.ctime()+' - Target set to '+self.__data.getTarget()+' and data loaded from '+filename)

    def randomizeTarget(self):
        choice = randint(0, len(self.__data.getValidWavelengths())-1)
        target = self.__data.getValidWavelengths()[choice]
        self.__data.setTarget(target)
        logging.info(time.ctime()+' - Now training on '+target)
        print(time.ctime()+' - Now training on '+target)

    def recreateWindow(self, conv_width=24, predictions=6):
        self.__data._createWindow(conv_width=conv_width, predictions=predictions)
        logging.info(time.ctime()+' - New window created with conv_width='+str(conv_width)+' and predictions='+str(predictions))
        print(time.ctime()+' - New window created with conv_width='+str(conv_width)+' and predictions='+str(predictions))

    def train(self, maxEpochs=2*5000, totalPatience=None):
        if totalPatience != None and isinstance(totalPatience, int):
            logging.info(time.ctime()+' - Now training on '+self.__data.getTarget()+' with max epochs: '+str(maxEpochs)+' and total patience: '+str(totalPatience))
            print(time.ctime()+' - Now training on '+self.__data.getTarget()+' with max epochs: '+str(maxEpochs)+' and total patience: '+str(totalPatience))
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=int(totalPatience),
                mode='min',
            )
        else:
            logging.info(time.ctime()+' - Now training on '+self.__data.getTarget()+' with max epochs: '+str(maxEpochs)+' and total patience: '+str(self.__data.getNumFeatures()))
            print(time.ctime()+' - Now training on '+self.__data.getTarget()+' with max epochs: '+str(maxEpochs)+' and total patience: '+str(self.__data.getNumFeatures()))
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=int(self.__data.getNumFeatures()),
                mode='min',
            )

        auto_save = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            )

        history = self.__model.fit(self.__data.getWindowTrainData(), 
                                   epochs=maxEpochs, 
                                   validation_data=self.__data.getWindowTrainValidation(),
                                   callbacks=[early_stopping, auto_save],# 
                                   )

        self._saveModel()
        logging.info(time.ctime()+' - Completed training on '+self.__data.getTarget())
        print(time.ctime()+' - Completed training on '+self.__data.getTarget())
        return history
        
    def evaluate(self):
        val_performance, performance = {}, {}
        val_performance['Residual LSTM'] = self.__model.evaluate(self.__data.getWindowTrainValidation(), return_dict=True)
        performance['Residual LSTM'] = self.__model.evaluate(self.__data.getWindowTrainTest(), verbose=0, return_dict=True)
        x = np.arange(len(performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        val_mae = [v[metric_name] for v in val_performance.values()]
        test_mae = [v[metric_name] for v in performance.values()]
        plt.figure(figsize=(16, 9))
        plt.ylabel(f'mean_absolute_error [{self.__data.getTarget()}, normalized]')
        plt.bar(x - 0.17, val_mae, width, label='Validation Set')
        plt.bar(x + 0.17, test_mae, width, label='Test Set')
        plt.gcf().suptitle(f'Model Performance of {self.__data.getTarget()} vs Validation Set and Test Set')
        plt.xticks(ticks=x, labels=performance.keys(),
                rotation=30)
        _ = plt.legend()
        plt.tight_layout()
        plt.savefig('graphs/'+'performance_'+self.__data.getTarget()+'.png')
        self.__data.plotModel(self.__model) # Plots the previously assigned Target Column
        # plt.show()
        logging.info(time.ctime()+' - New graphs generated in ./graph/ folder:\n'+'graphs/'+'performance_'+self.__data.getTarget()+'.png\n'+'graphs/'+self.__data.getTarget()+'.png')
        print(time.ctime()+' - New graphs generated in ./graph/ folder:\n'+'graphs/'+'performance_'+self.__data.getTarget()+'.png\n'+'graphs/'+self.__data.getTarget()+'.png')
# end Arbiter