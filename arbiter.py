### Developed by Anthony Castillo ###
# https://www.tensorflow.org/install/pip
# https://www.tensorflow.org/guide/gpu
import time
import logging
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from dataHandler import DataHandler
from residualWrapper import ResidualWrapper

class Arbiter():
    def __init__(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # tf.keras.utils.get_custom_objects().update({'ResidualWrapper': ResidualWrapper})
        self._configureDirectories()
        self.__model = None
        self.__modelName = 'Arbiter'
        self.__data = DataHandler()
        self.__val_performance, self.__performance = {}, {}
    # end __init__

    def __del__(self):
        del self.__model, self.__modelName, self.__data, self.__val_performance, self.__performance
    # end __del__

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
    # end _configureFilename

    def _setLogger(self):
        self.__currTime = time.localtime()
        logging.basicConfig(filename='./logs/'+self._configureFilename(self.__currTime)+'.log', encoding='utf-8', level=logging.DEBUG)
        logging.info(time.ctime()+' - Initializing...')
        logging.info(time.ctime()+' - Saving log to runtime_'+self._configureFilename(self.__currTime)+'.log')
        print(time.ctime()+' - Saving log to runtime_'+self._configureFilename(self.__currTime)+'.log')
    # end _setLogger

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
    # end _configureDirectories

    def _createModel(self):#, conv_width=24, predictions=24):
        """Creates a new residual long short-term memory model
        """
        # print(tf.keras.config.floatx())
        # tf.keras.backend.set_floatx('float64')
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
        self.compile()
        logging.info(time.ctime()+' - New model created.')
    # end _createModel
    
    def compile(self):
        # tf.keras.backend.set_floatx('float64')
        self.__model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
    # end compile

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
    # end _saveModel

    def readModel(self, conv_width:int=24, predictions:int=6):
        """Reads a model from file using the name provided in self.__modelName
        """
        self.__data.createWindow(conv_width=conv_width, predictions=predictions, label_columns=False)
        try:
            self.__model = ResidualWrapper(tf.keras.models.load_model('models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras'))#, custom_objects={'ResidualWrapper': ResidualWrapper})
            self.compile()
            logging.info(time.ctime()+' - Found and loaded model '+'models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            print(time.ctime()+' - Found and loaded model '+'models/'+self.__modelName+'_'+self.__data.getTarget()+'.keras')
            self.evaluate()
        except Exception as error1:
            try:
                logging.info(time.ctime()+' - '+str(error1))
                print(error1)
                self.__model = ResidualWrapper(tf.keras.models.load_model('models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras'))#, custom_objects={'ResidualWrapper': ResidualWrapper})
                self.compile()
                logging.info(time.ctime()+' - Found and loaded model '+'models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras')
                print(time.ctime()+' - Found and loaded model '+'models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras')
                self.evaluate()
            except Exception as error2:
                logging.info(time.ctime()+' - '+str(error1))
                print(error2)
                logging.info(time.ctime()+' - Colud not find model for '+self.__data.getTarget())
                print(time.ctime()+' - Colud not find model for '+self.__data.getTarget())
                self._createModel()
    # end readModel

    def loadData(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        self.__data.readDataFromFile(filename, format)
        self.__data.setTarget()
        logging.info(time.ctime()+' - Target set to '+self.__data.getTarget()+' and data loaded from '+filename)
        print(time.ctime()+' - Target set to '+self.__data.getTarget()+' and data loaded from '+filename)
    # end loadData

    def randomizeTarget(self):
        choice = randint(0, len(self.__data.getValidWavelengths())-1)
        target = self.__data.getValidWavelengths()[choice]
        self.__data.setTarget(target)
        logging.info(time.ctime()+' - Now training on '+target)
        print(time.ctime()+' - Now training on '+target)
    # end randomizeTarget

    def shiftTarget(self):
        if self.__data.getValidWavelengths().index(self.__data.getTarget()) != len(self.__data.getValidWavelengths())-1:

            self.__data.setTarget(self.__data.getValidWavelengths()[self.__data.getValidWavelengths().index(self.__data.getTarget())+1]) # Increment to the next wavelength
        else:
            self.__data.setTarget(self.__data.getValidWavelengths()[0])
        logging.info(time.ctime()+' - Now training on '+self.__data.getTarget())
        print(time.ctime()+' - Now training on '+self.__data.getTarget())
    # end shiftTarget

    def getTarget(self):
        return self.__data.getTarget()
    # end getTarget

    def getValidWavelengths(self):
        return self.__data.getValidWavelengths()
    # end getValidWavelengths

    def recreateWindow(self, conv_width:int=24, predictions:int=6):
        self.__data._createWindow(conv_width=conv_width, predictions=predictions)
        logging.info(time.ctime()+' - New window created with conv_width='+str(conv_width)+' and predictions='+str(predictions))
        print(time.ctime()+' - New window created with conv_width='+str(conv_width)+' and predictions='+str(predictions))
    # end recreateWindow

    def train(self, maxEpochs:int=2*5000, totalPatience:int=None):
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
        history = self.__model.fit(
            self.__data.getWindowTrainData(), 
            epochs=maxEpochs, 
            validation_data=self.__data.getWindowTrainValidation(),
            callbacks=[early_stopping, auto_save],# 
        )
        self.__model = ResidualWrapper(tf.keras.models.load_model('models/checkpoints/'+self.__modelName+'_'+self.__data.getTarget()+'_checkpoint.keras'))
        self.compile()
        self._saveModel()
        logging.info(time.ctime()+' - Completed training on '+self.__data.getTarget())
        print(time.ctime()+' - Completed training on '+self.__data.getTarget())
        return history
    # end train

    def evaluate(self):
        self.__val_performance[self.__data.getTarget()] = self.__model.evaluate(self.__data.getWindowTrainValidation(), return_dict=True)
        self.__performance[self.__data.getTarget()] = self.__model.evaluate(self.__data.getWindowTrainTest(), verbose=0, return_dict=True)
        x = np.arange(len(self.__performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        val_mae = [v[metric_name] for v in self.__val_performance.values()]
        test_mae = [v[metric_name] for v in self.__performance.values()]
        plt.figure(figsize=(16, 9))
        plt.ylabel('mean_absolute_error')
        plt.bar(x - 0.17, val_mae, width, label='Validation Set')
        plt.bar(x + 0.17, test_mae, width, label='Test Set')
        plt.gcf().suptitle('Model Performance of the Residual LSTM model vs Validation Set and Test Set')
        plt.xticks(ticks=x, labels=self.__performance.keys(), rotation=30)
        _ = plt.legend()
        plt.tight_layout()
        plt.savefig('graphs/'+'performance_residual_lstm.png')
        self.__data.plotModel(self.__model) # Plots the previously assigned Target Column
        # plt.show()
        logging.info(time.ctime()+' - New graphs generated in ./graph/ folder:\n\t'+'graphs/'+'performance_residual_lstm.png\n\t'+'graphs/'+self.__data.getTarget()+'.png')
        print(time.ctime()+' - New graphs generated in ./graph/ folder:\n\t'+'graphs/'+'performance_residual_lstm.png\n\t'+'graphs/'+self.__data.getTarget()+'.png')
    # end evaluate
# end Arbiter