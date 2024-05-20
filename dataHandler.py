### Developed by Anthony Castillo ###
### Refer to this page: https://www.tensorflow.org/tutorials/structured_data/time_series ###
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import tensorflow as tf
import seaborn as sns
from windowGenerator import WindowGenerator
from baseline import Baseline
from residualWrapper import ResidualWrapper
import IPython
import IPython.display
from threading import Thread
import time

class DataHandler():
    def __init__(self):
        self.__siteName = ''
        self.__data = pd.DataFrame()
        self.__normalizedData = None
        self.__datetime = None
        self.__data_std = None
        self.__validWavelengthCount = 0
        self.__validWavelengths = []
        self.__AODTotalColumns = []
        self.__threads = []
        self.__num_features = 0
        self.__window = None
        self.__target = None

    def __del__(self):
        pass

    def getSiteName(self):
        return self.__siteName
    
    def getData(self):
        return self.__data
    
    def getNormalizedData(self):
        return self.__normalizedData
    
    def getValidWavelengths(self):
        return self.__validWavelengths
    
    def setData(self, data):
        self.__data = data

    def getWindowTrainData(self):
        return self.__window.train
    
    def getWindowTrainValidation(self):
        return self.__window.val
    
    def getWindowTrainTest(self):
        return self.__window.test
    
    def getNumFeatures(self):
        return self.__num_features
    
    def createWindow(self, conv_width=24, predictions=6):
        self._createWindow(conv_width=conv_width, predictions=predictions)

    def setTarget(self, target:str=''):
        if target != '':
            self.__target = target
        else:
            if len(self.__validWavelengths) > 0:
                print('Select Target:\n')
                choices = []
                for i, v in enumerate(self.__validWavelengths):
                    choices.append(str(i))
                    print('['+str(i)+']\t'+str(v))
                userInput = input('Choice: ')
                if userInput in choices:
                    self.__target = self.__validWavelengths[int(userInput)]
                    print('Selected '+self.__validWavelengths[int(userInput)]+'.')
                else:
                    print('Please select a valid target.')
                    self.setTarget(target)
            else: # Default setting
                print('Valid Wavelenghts not detected, using default value of AOD_380nm-Total.')
                self.__target = 'AOD_380nm-Total'

    def getTarget(self):
        return self.__target
    
    def plotModel(self, model):
        print(model)
        # for i in self.__validWavelengths:#self.__AODTotalColumns:#self.__target:#
            # print(i)
            # self.__window.plot(model, plot_col=i)
        self.__window.plot(model, plot_col=self.__target)
        plt.show()

    def readDataFromFile(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        """Reads data from the given file and begins processing it.

        Args:
            filename (str, optional): The name of the file (csv) containing the data. Defaults to 'data/20230101_20241231_Turlock_CA_USA.tot_lev15'.
            format (str, optional): Unused parameter. Defaults to 'csv'.
        """
        # self.__target=target
        # pd.set_option('display.max_rows', 300, 'display.max_columns', 300)
        # Retriving just the sitename of the data
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for line in csvreader:
                self.__siteName = line[0]
                break

        # Getting data from csv file, processing the datetime of extraction, and droping specific columns
        self.__data = pd.read_csv(filename,skiprows=6, encoding='utf-8')#, parse_dates={'datetime':[0,1]})
        # self.__data['time_of_day'] = pd.to_datetime(self.__data['Time(hh:mm:ss)'], format='%H:%M:%S').dt.as_unit('s') # Supported units are 's', 'ms', 'us', 'ns'
        self.__data['datetime'] = self.__data['Date(dd:mm:yyyy)'] + ' ' + self.__data['Time(hh:mm:ss)']
        self.__data = self.__data.drop(columns=['Date(dd:mm:yyyy)', 'Time(hh:mm:ss)', 'Data_Quality_Level', 'AERONET_Site_Name', 'Last_Date_Processed']) # Dropping these columns due to their dtype being of a string nature
        # self.__data['datetime'] = pd.to_datetime(self.__data['datetime'], format='%d:%m:%Y %H:%M:%S')
        self.__datetime = pd.to_datetime(self.__data.pop('datetime'), format='%d:%m:%Y %H:%M:%S')

        #Replacing -999 with either a 0 or the mean value for the column
        for i, iWaveLength in enumerate(self.__data.columns):
            if iWaveLength == 'datetime':
                continue
            if 'Total' in iWaveLength and 'AOD' in iWaveLength:
            # if 'AOD' in iWaveLength and 'nm' in iWaveLength and 'um' not in iWaveLength and 'Rayleigh' not in iWaveLength:
                self.__AODTotalColumns.append(i)
            self.__data[iWaveLength] = self.__data[iWaveLength].replace(-999., np.nan).astype(np.float32) # Setting to Uniform dypes
            if pd.isna(self.__data[iWaveLength].mean()):
                self.__data[iWaveLength] = self.__data[iWaveLength].fillna(0) # Setting nan values to 0
            else:
                self.__data[iWaveLength] = self.__data[iWaveLength].fillna(self.__data[iWaveLength].mean()) # Setting values to mean of column
            
        # print(self.__data.isnull().any()) # True means NULL is in the column

        # # Getting Valid Wavelengths
        # self.__validWavelengthCount = 0
        # for i in self.__data.columns[self.__AODTotalColumns]:
        #     if(self.__data[i].mean() > 0):
        #         self.__validWavelengthCount += 1

        
        # Info, graph, and normalize
        # print(self.__data.describe().transpose())
        plot_cols = [] #['AOD_1640nm-Total', 'AOD_412nm-Total', 'AOD_340nm-Total']
        self.__validWavelengthCount = 0
        for i in self.__data.columns[self.__AODTotalColumns]:
            if(self.__data[i].mean() > 0):
                self.__validWavelengthCount += 1
                plot_cols.append(i)
        if self.__validWavelengths != plot_cols:
            self.__validWavelengths = plot_cols
        print('Valid Wavelengths: '+str(self.__validWavelengths))
        self._graphData()
        self._NormalizeData()
        # self._createWindow(conv_width=conv_width, predictions=predictions)
        # self._window_test()
        # self._multiWindowTest()
    # end readDataFromFile

    def _graphData(self):
        """Generates a graph showing the availabilty of the data.

        Args:
            filename (str): _description_
        """
        # # Getting Valid Wavelengths
        
        
        plot_features = self.__data[self.__validWavelengths]
        plot_features.index = self.__datetime
        plots1 = plot_features.plot(subplots=True)

        # plot_features = self.__data[self.__validWavelengths][:480]
        # plot_features.index = self.__datetime[:480]
        # plots2 = plot_features.plot(subplots=True)

        # ## Formatting the graph
        plt.gcf().autofmt_xdate()
        plt.gcf().suptitle('Valid Wavelenghts')
        # plt.title()
        # plt.grid(which='major',axis='both')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.tight_layout()
        # plt.show()

        # Creating a time of day signal for the model to utilize
        timestamp_s = self.__datetime.map(pd.Timestamp.timestamp)
        day = 24*60*60
        # self.__data['Day_Sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        # self.__data['Day_Cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        newColumns = pd.DataFrame({
            'Day_Sin': np.sin(timestamp_s * (2 * np.pi / day)),
            'Day_Cos': np.cos(timestamp_s * (2 * np.pi / day)),
            })
        self.__data = pd.concat([self.__data, newColumns], axis=1)
        # plt.figure(figsize=(12, 8))
        # plt.plot(np.array(self.__data['Day_Sin'])[:500])
        # plt.plot(np.array(self.__data['Day_Cos'])[:500])
        # plt.xlabel('Time [h]')
        # plt.title('Time of day signal')
        # plt.tight_layout()
        # plt.show()
    # end _graphData

    def _NormalizeData(self):
        column_indicies = {name: i for i, name, in enumerate(self.__data.columns)}
        n=len(self.__data)
        train_df = self.__data[0:int(n*0.7)]
        val_df = self.__data[int(n*0.7):int(n*0.9)]
        test_df = self.__data[int(n*0.9):]
        self.__num_features = self.__data.shape[1]
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        self.__normalizedData = (train_df, val_df, test_df,)

        self.__data_std = (self.__data - train_mean) / train_std
        self.__data_std = self.__data_std.melt(var_name='Column', value_name='Normalized')
        # plt.figure(figsize=(12,6))
        # ax = sns.violinplot(x='Column', y='Normalized', data=self.__data_std[:20000*5])
        # plot0 = ax.set_xticklabels(self.__data.keys(), rotation=45)
        # plt.show()
    # end _NormalizeData

    def _createWindow(self, conv_width=24, predictions=6):
        self.__window = WindowGenerator(
            input_width=conv_width,#self.__validWavelengthCount,
            label_width=1,#conv_width,#predictions,#self.__validWavelengthCount,
            shift=1,#1,#predictions,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            label_columns=[self.__target],#self.__validWavelengths,#self.__AODTotalColumns
        )
    # end _createWindow

    def _window_test(self):
        w1 = WindowGenerator(input_width=24, label_width=1, shift=24,train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
                     label_columns=self.__validWavelengths[-2:-1])
        print(w1)

        w2 = WindowGenerator(input_width=6, label_width=1, shift=1, train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
                     label_columns=self.__validWavelengths[-2:-1])
        print(w2)

        example_window = tf.stack([np.array(self.__normalizedData[0][:w2.total_window_size]), 
                                np.array(self.__normalizedData[0][100:100+w2.total_window_size]), 
                                np.array(self.__normalizedData[0][200:200+w2.total_window_size])])
        example_inputs, example_labels = w2.split_window(example_window)
        print('All shapes are: (batch, time, features)')
        print(f'Window shape: {example_window.shape}')
        print(f'Inputs shape: {example_inputs.shape}')
        print(f'Labels shape: {example_labels.shape}')
        w2.plot()
        # w2.example = example_inputs, example_labels
        w2.plot(plot_col=self.__validWavelengths[-3])

        # Each element is an (inputs, label) pair.
        print(w2.train.element_spec)

        for example_inputs, example_labels in w2.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')

        single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
            label_columns=['AOD_380nm-Total'])
        print(single_step_window)

        for example_inputs, example_labels in single_step_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
            
        baseline = Baseline(label_index=w2.getColumnIndicies()['AOD_380nm-Total'])

        baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        val_performance = {}
        performance = {}
        val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
        performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

        wide_window = WindowGenerator(
            input_width=24, label_width=24, shift=1, train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
            label_columns=['AOD_380nm-Total'])

        print(wide_window)
        print('Input shape:', wide_window.example[0].shape)
        print('Output shape:', baseline(wide_window.example[0]).shape)
        wide_window.plot(baseline)
        plt.show()

        linear = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1)
            ])

        print('Input shape:', single_step_window.example[0].shape)
        print('Output shape:', linear(single_step_window.example[0]).shape)

        # compile_and_fit
        # print(type(linear), type(single_step_window))
        history = self.compile_and_fit(linear, single_step_window)

        val_performance['Linear'] = linear.evaluate(single_step_window.val, return_dict=True)
        performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)

        print('Input shape:', wide_window.example[0].shape)
        print('Output shape:', linear(wide_window.example[0]).shape)

        wide_window.plot(linear)
        # plt.show()
        plt.bar(x = range(len(self.__normalizedData[0].columns)),
                height=linear.layers[0].kernel[:,0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(self.__normalizedData[0].columns)))
        _ = axis.set_xticklabels(self.__normalizedData[0].columns, rotation=90)
        plt.show()

        dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

        history = self.compile_and_fit(dense, single_step_window)

        val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
        performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

        CONV_WIDTH = 3
        conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            label_columns=['AOD_380nm-Total'])

        conv_window
        conv_window.plot()
        plt.suptitle("Given 3 hours of inputs, predict 1 hour into the future.")
        plt.show()

        multi_step_dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])

        print('Input shape:', conv_window.example[0].shape)
        print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

        history = self.compile_and_fit(multi_step_dense, conv_window)

        IPython.display.clear_output()
        val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val, return_dict=True)
        performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0, return_dict=True)
        print('Multi step dense')
        # plt.figure(figsize=(12, 8))
        conv_window.plot(multi_step_dense)
        plt.show()
        
        print('Input shape:', wide_window.example[0].shape)
        try:
            print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
        except Exception as e:
            print(f'\n{type(e).__name__}:{e}')

        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                kernel_size=(CONV_WIDTH,),
                                activation='relu'),
            # tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
        ])

        print("Conv model on `conv_window`")
        print('Input shape:', conv_window.example[0].shape)
        print('Output shape:', conv_model(conv_window.example[0]).shape)

        history = self.compile_and_fit(conv_model, conv_window)

        IPython.display.clear_output()
        val_performance['Conv'] = conv_model.evaluate(conv_window.val, return_dict=True)
        performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)

        print("Wide window")
        print('Input shape:', wide_window.example[0].shape)
        print('Labels shape:', wide_window.example[1].shape)
        print('Output shape:', conv_model(wide_window.example[0]).shape)

        LABEL_WIDTH = 24
        INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
        wide_conv_window = WindowGenerator(
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            label_columns=['AOD_380nm-Total'])

        print(wide_conv_window)

        print("Wide conv window")
        print('Input shape:', wide_conv_window.example[0].shape)
        print('Labels shape:', wide_conv_window.example[1].shape)
        print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

        wide_conv_window.plot(conv_model)
        plt.show()

        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])

        print('Input shape:', wide_window.example[0].shape)
        print('Output shape:', lstm_model(wide_window.example[0]).shape)

        history = self.compile_and_fit(lstm_model, wide_window)

        IPython.display.clear_output()
        val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, return_dict=True)
        performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

        wide_window.plot(lstm_model)
        plt.show()

        cm = lstm_model.metrics[1]
        print(cm.metrics)

        print(val_performance)

        x = np.arange(len(performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        val_mae = [v[metric_name] for v in val_performance.values()]
        test_mae = [v[metric_name] for v in performance.values()]

        plt.figure(figsize=(12, 8))
        plt.ylabel('mean_absolute_error [T (degC), normalized]')
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=performance.keys(),
                rotation=45)
        _ = plt.legend()
        # plt.show()

        for name, value in performance.items():
            print(f'{name:12s}: {value[metric_name]:0.4f}')

        single_step_window = WindowGenerator(
            # `WindowGenerator` returns all features as labels if you
            # don't set the `label_columns` argument.
            input_width=1, label_width=1, shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            )

        wide_window = WindowGenerator(
            input_width=24, label_width=24, shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            )

        for example_inputs, example_labels in wide_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')

        baseline = Baseline()
        baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        val_performance = {}
        performance = {}
        val_performance['Baseline'] = baseline.evaluate(wide_window.val, return_dict=True)
        performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0, return_dict=True)

        dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=self.__num_features)
        ])

        history = self.compile_and_fit(dense, single_step_window)

        IPython.display.clear_output()
        val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
        performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

        wide_window = WindowGenerator(
            input_width=24, label_width=24, shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            )

        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=self.__num_features)
        ])

        history = self.compile_and_fit(lstm_model, wide_window)

        IPython.display.clear_output()
        val_performance['LSTM'] = lstm_model.evaluate( wide_window.val, return_dict=True)
        performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0, return_dict=True)

        # print()
    # end window

    def _multiWindowTest(self):
        single_step_window = WindowGenerator(
            # `WindowGenerator` returns all features as labels if you
            # don't set the `label_columns` argument.
            input_width=1,
            label_width=1,
            shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            )

        wide_window = WindowGenerator(
            input_width=24,
            label_width=24,
            shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            )

        for example_inputs, example_labels in wide_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        
        # Baseline

        baseline = Baseline()
        baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        val_performance = {}
        performance = {}
        val_performance['Baseline'] = baseline.evaluate(wide_window.val, return_dict=True)
        performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0, return_dict=True)

        # Dense

        dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=self.__num_features)
        ])

        history = self.compile_and_fit(dense, single_step_window)

        IPython.display.clear_output()
        val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
        performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

        wide_window = WindowGenerator(
            input_width=24, label_width=24, shift=1,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
            )

        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=self.__num_features)
        ])

        history = self.compile_and_fit(lstm_model, wide_window)

        IPython.display.clear_output()
        val_performance['LSTM'] = lstm_model.evaluate( wide_window.val, return_dict=True)
        performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0, return_dict=True)

        residual_lstm = ResidualWrapper(
            tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                self.__num_features,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros())
        ]))

        history = self.compile_and_fit(residual_lstm, wide_window)

        IPython.display.clear_output()
        val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val, return_dict=True)
        performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0, return_dict=True)

        x = np.arange(len(performance))
        width = 0.3

        metric_name = 'mean_absolute_error'
        val_mae = [v[metric_name] for v in val_performance.values()]
        test_mae = [v[metric_name] for v in performance.values()]

        plt.figure(figsize=(12, 8))
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=performance.keys(),
                rotation=45)
        plt.ylabel('MAE (average over all outputs)')
        _ = plt.legend()
        plt.show()

        for name, value in performance.items():
            print(f'{name:15s}: {value[metric_name]:0.4f}')
    # end _multiWindowTest

    def compile_and_fit(self, model, window, patience=2, MAX_EPOCHS=20):
        # print(type(model), type(window), type(patience))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min') # 'val_accuracy',
        
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()]) # 'accuracy'])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history
    # end compile_and_fit
# end DataHandler