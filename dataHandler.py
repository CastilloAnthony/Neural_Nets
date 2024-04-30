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

    def readDataFromFile(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        """Reads data from the given file and begins processing it.

        Args:
            filename (str, optional): The name of the file (csv) containing the data. Defaults to 'data/20230101_20241231_Turlock_CA_USA.tot_lev15'.
            format (str, optional): Unused parameter. Defaults to 'csv'.
        """
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
        self._graphData('')
        self._NormalizeData()
        self._window_test()
    # end readDataFromFile

    def _graphData(self, filename:str):
        """Generates a graph showing the availabilty of the data.

        Args:
            filename (str): _description_
        """
        # # Getting Valid Wavelengths
        plot_cols = [] #['AOD_1640nm-Total', 'AOD_412nm-Total', 'AOD_340nm-Total']
        self.__validWavelengthCount = 0
        for i in self.__data.columns[self.__AODTotalColumns]:
            if(self.__data[i].mean() > 0):
                self.__validWavelengthCount += 1
                plot_cols.append(i)
        self.__validWavelengths = plot_cols
        
        plot_features = self.__data[plot_cols]
        plot_features.index = self.__datetime
        plots1 = plot_features.plot(subplots=True)

        plot_features = self.__data[plot_cols][:480]
        plot_features.index = self.__datetime[:480]
        plots2 = plot_features.plot(subplots=True)

        # ## Formatting the graph
        plt.gcf().autofmt_xdate()
        # plt.grid(which='major',axis='both')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.tight_layout()
        # plt.show()

        timestamp_s = self.__datetime.map(pd.Timestamp.timestamp)
        day = 24*60*60
        self.__data['Day_Sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.__data['Day_Cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        plt.plot(np.array(self.__data['Day_Sin'])[:500])
        plt.plot(np.array(self.__data['Day_Cos'])[:500])
        plt.xlabel('Time [h]')
        plt.title('Time of day signal')
        plt.tight_layout()
        # plt.show()
    # end _graphData

    def _NormalizeData(self):
        column_indicies = {name: i for i, name, in enumerate(self.__data.columns)}
        n=len(self.__data)
        train_df = self.__data[0:int(n*0.7)]
        val_df = self.__data[int(n*0.7):int(n*0.9)]
        test_df = self.__data[int(n*0.9):]
        num_features = self.__data.shape[1]
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        self.__normalizedData = (train_df, val_df, test_df,)

        self.__data_std = (self.__data - train_mean) / train_std
        self.__data_std = self.__data_std.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12,6))
        ax = sns.violinplot(x='Column', y='Normalized', data=self.__data_std[:20000*5])
        plot0 = ax.set_xticklabels(self.__data.keys(), rotation=45)
        # plt.show()
    # end _NormalizeData

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

        linear = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1)
            ])

        print('Input shape:', single_step_window.example[0].shape)
        print('Output shape:', linear(single_step_window.example[0]).shape)

        # compile_and_fit
        print(type(linear))
        history = self.compile_and_fit(model=linear, window=single_step_window)

        val_performance['Linear'] = linear.evaluate(single_step_window.val, return_dict=True)
        performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)

        print('Input shape:', wide_window.example[0].shape)
        print('Output shape:', linear(wide_window.example[0]).shape)

        wide_window.plot(linear)

        plt.bar(x = range(len(self.__normalizedData[0].columns)),
        height=linear.layers[0].kernel[:,0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(self.__normalizedData[0].columns)))
        _ = axis.set_xticklabels(self.__normalizedData[0].columns, rotation=90)
        plt.show()
    # end window

    def compile_and_fit(model, window, patience=2, MAX_EPOCHS=20):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min') # 'val_accuracy',
        print(type(model))
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()]) # 'accuracy'])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history
    # end compile_and_fit
# end DataHandler