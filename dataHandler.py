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
import IPython
import IPython.display
from threading import Thread
import time
from windowGenerator import WindowGenerator
from baseline import Baseline
from residualWrapper import ResidualWrapper
from window_test import window_test, multiWindowTest, compile_and_fit

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
    
    def createWindow(self, conv_width=24, predictions=24, label_columns=True):
        self._createWindow(conv_width=conv_width, predictions=predictions, label_columns=label_columns)

    def setTarget(self, target:str=''):
        if target != '':
            self.__target = target
        else:
            if len(self.__validWavelengths) > 0:
                print('\nSelect Target:')
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
        print()

    def getTarget(self):
        return self.__target
    
    def getTargetIndex(self):
        return self.__window.getColumnIndicies()[self.__target]

    def plotModel(self, model):
        # print(model)
        # for i in self.__validWavelengths:#self.__AODTotalColumns:#self.__target:#
            # print(i)
            # self.__window.plot(model, plot_col=i)
        self.__window.plot(model=model, plot_col=self.__target, max_subplots=3)# 
        # plt.show()

    def readDataFromFile(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        """Reads data from the given file and begins processing it.

        Args:
            filename (str, optional): The name of the file (csv) containing the data. Defaults to 'data/20230101_20241231_Turlock_CA_USA.tot_lev15'.
            format (str, optional): Unused parameter. Defaults to 'csv'.
        """
        # Retriving just the sitename of the data
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for line in csvreader:
                self.__siteName = line[0]
                break

        # Getting data from csv file, processing the datetime of extraction, and droping specific columns
        self.__data = pd.read_csv(filename,skiprows=6, encoding='utf-8')#, parse_dates={'datetime':[0,1]})
        self.__data['datetime'] = self.__data['Date(dd:mm:yyyy)'] + ' ' + self.__data['Time(hh:mm:ss)']
        self.__data = self.__data.drop(columns=['Date(dd:mm:yyyy)', 'Time(hh:mm:ss)', 'Data_Quality_Level', 'AERONET_Site_Name', 'Last_Date_Processed']) # Dropping these columns due to their dtype being of a string nature
        self.__datetime = pd.to_datetime(self.__data.pop('datetime'), format='%d:%m:%Y %H:%M:%S')

        #Replacing -999 with either a 0 or the mean value for the column
        for i, iWaveLength in enumerate(self.__data.columns):
            if iWaveLength == 'datetime':
                continue
            if 'Total' in iWaveLength and 'AOD' in iWaveLength:
                self.__AODTotalColumns.append(i)
            self.__data[iWaveLength] = self.__data[iWaveLength].replace(-999., np.nan).astype(np.float32) # Setting to Uniform dypes
            if pd.isna(self.__data[iWaveLength].mean()):
                self.__data[iWaveLength] = self.__data[iWaveLength].fillna(0) # Setting nan values to 0
            else:
                self.__data[iWaveLength] = self.__data[iWaveLength].fillna(self.__data[iWaveLength].mean()) # Setting values to mean of column
            
        plot_cols = [] #['AOD_1640nm-Total', 'AOD_412nm-Total', 'AOD_340nm-Total']
        self.__validWavelengthCount = 0
        for i in self.__data.columns[self.__AODTotalColumns]:
            if(self.__data[i].mean() > 0):
                self.__validWavelengthCount += 1
                plot_cols.append(i)
        if self.__validWavelengths != plot_cols:
            self.__validWavelengths = plot_cols
        print('Detected Valid Wavelengths: '+str(self.__validWavelengths))

        # Creating a time of day signal for the model to utilize
        timestamp_s = self.__datetime.map(pd.Timestamp.timestamp)
        day = 24*60*60
        newColumns = pd.DataFrame({
            'Day_Sin': np.sin(timestamp_s * (2 * np.pi / day)),
            'Day_Cos': np.cos(timestamp_s * (2 * np.pi / day)),
            })
        self.__data = pd.concat([self.__data, newColumns], axis=1)

        self._graphData()
        self._NormalizeData()
        # self._createWindow(conv_width=24, predictions=6)

        ### Non-functional test environments
        # window_test()
        # multiWindowTest()
    # end readDataFromFile

    def _graphData(self):
        """Generates a graph showing the availabilty of the data.
        """
        ### Plotting Valid Wavelengths
        plot_features = self.__data[self.__validWavelengths]
        plot_features.index = self.__datetime
        plots1 = plot_features.plot(subplots=True, figsize=(16, 9))
        plt.gcf().autofmt_xdate()
        plt.gcf().suptitle('Valid Wavelenghts')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.tight_layout()
        plt.savefig('graphs/data_availability.png')
        # plt.show()
        
        ### Plotting Time of day signal
        plt.figure(figsize=(16, 9))
        plt.plot(np.array(self.__data['Day_Sin'])[:500])
        plt.plot(np.array(self.__data['Day_Cos'])[:500])
        plt.xlabel('Time [h]')
        plt.title('Time of day signal')
        plt.grid(which='major',axis='y')
        plt.tight_layout()
        plt.savefig('graphs/data_time_signal.png')
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

        # Normalize Data ## Caused an over normalized data set issue with NAN, Zero, and near Zero values - DO NOT USE
        # train_df = (train_df - train_mean) / train_std
        # val_df = (val_df - train_mean) / train_std
        # test_df = (test_df - train_mean) / train_std
        self.__normalizedData = (train_df, val_df, test_df,)

        self.__data_std = (self.__data[self.__validWavelengths])# - train_mean) / train_std
        # self.__data_std = self.__data_std.melt(var_name='Column', value_name='Normalized')
        self.__data_std = self.__data_std.melt(var_name='Valid Wavelengths', value_name='Normalized')
        plt.figure(figsize=(16,9))
        # sns.set_theme(rc={'figure.figsize':(16,9)})
        ax = sns.violinplot(x='Valid Wavelengths', y='Normalized', data=self.__data_std[:20000*(len(self.__validWavelengths)+1)], native_scale=True)
        plot0 = ax.set_xticklabels(self.__validWavelengths, rotation=30)#self.__data_std.keys(), rotation=45)
        plt.title('Distribution of Valid Wavelength Data')
        plt.grid(which='major',axis='y')
        plt.tight_layout()
        plt.savefig('graphs/data_violin.png')
        # plt.show()
    # end _NormalizeData

    def _createWindow(self, conv_width=24, predictions=24, label_columns=True):
        if label_columns == True:
            self.__window = WindowGenerator(
                input_width=conv_width,#+predictions-1,#self.__validWavelengthCount,
                label_width=predictions,#predictions,#self.__validWavelengthCount,
                shift=1,#1,#predictions,
                train_df=self.__normalizedData[0], 
                val_df=self.__normalizedData[1], 
                test_df=self.__normalizedData[2],
                label_columns=[self.__target],#self.__validWavelengths,#self.__AODTotalColumns
            )
        else:
            self.__window = WindowGenerator(
            input_width=conv_width,#+predictions-1,#self.__validWavelengthCount,
            label_width=predictions,#predictions,#self.__validWavelengthCount,
            shift=1,#1,#predictions,
            train_df=self.__normalizedData[0], 
            val_df=self.__normalizedData[1], 
            test_df=self.__normalizedData[2],
        )
    # end _createWindow