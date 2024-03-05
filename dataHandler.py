import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from tensorflow import convert_to_tensor, float64

class DataHandler():
    def __init__(self):
        self.__siteName = ''
        self.__data = pd.DataFrame()
        self.__tensorData = None
        self.__validWavelengthCount = 0
        self.__AODTotalColumns = []

    def __del__(self):
        pass

    def getSiteName(self):
        return self.__siteName
    
    def getData(self):
        return self.__data
    
    def getTensorData(self):
        return self.__tensorData
    
    def setData(self, data):
        self.__data = data

    def readDataFromFile(self, filename:str='data/20230101_20241231_Turlock_CA_USA.tot_lev15', format:str='csv'):
        # Retriving just the sitename of the data
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for line in csvreader:
                self.__siteName = line[0]
                break

        # Getting data from csv file
        self.__data = pd.read_csv(filename,skiprows=6, parse_dates={'datetime':[0,1]})
        # datetime_utc=pd.to_datetime(self.__data["datetime"], format='%d:%m:%Y %H:%M:%S')
        # datetime_pac= pd.to_datetime(datetime_utc).dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
        # self.__data.set_index(datetime_pac, inplace = True)
        print(self.__data.values)
        # for i in self.__data:
        #     print(self.__data[0])

        # Setting the Columns that has the AOD Total and replacing -999 with nan in all AOD_Total Wavelengths
        self.__AODTotalColumns=range(3,173,8)
        for iWaveLength in self.__data.columns[self.__AODTotalColumns]:
            self.__data[iWaveLength].replace(-999.0, np.nan, inplace = True)

        # Getting Valid Wavelengths
        self.__validWavelengthCount = 0
        for i in self.__data.columns[self.__AODTotalColumns]:
            if(self.__data[i].mean() > 0):
                self.__validWavelengthCount += 1

        # self._graphData(filename)
        self._convertDataToTensor()

    def _graphData(self, filename:str):
        """Generates a graph showing the availabilty of the data.

        Args:
            filename (str): _description_
        """
        StartDate='2016-01-01 00:00:00'
        EndDate='2023-12-31 23:59:59'

        ax = plt.figure(figsize=(16*.65,9*.65)).add_subplot(111) # 16:9 resolution scaled down to 65%
        ax.set_title(self.__siteName + ' Weekly AOD-Total Data Availability for ' + filename[14:18])

        ## Dynamically Adjusting Colors to increase the potential number of colors usable in the graph
        ### This should allow for all of the colors of the rainbow to be used, the higher number of valid wavelengths in our dataset, the more colors will be selected
        ### Note: The colors do not necessarily follow what their actual wavelengths are
        vaildWavelengthsCount = 0
        for i in self.__data.columns[self.__AODTotalColumns]:
            if(self.__data[i].mean() >= 0):
                vaildWavelengthsCount += 1
        cm = plt.get_cmap('gist_rainbow') # Color Mapping
        ax.set_prop_cycle(color=[cm(1.*i/vaildWavelengthsCount) for i in range(vaildWavelengthsCount)]) # Setting the color scheme to rainbow colors

        ## Adding the plots the graph
        count, handlesList = 0, []
        ### Comment out the next four lines to show only the AOD Data

        for iWaveLength in self.__data.columns[self.__AODTotalColumns]: 
            if(self.__data[iWaveLength].mean() > 0):
                dfGroup = self.__data.loc[StartDate:EndDate, iWaveLength].dropna().groupby([pd.Grouper(freq='W')]).size()
                dots = ax.plot(dfGroup[dfGroup <= 4*12*7]/(4*12*7)*100,'.',label=iWaveLength, markersize=vaildWavelengthsCount*3-count*2) # Excluding entries that will go above our 100% scale
                handlesList.append(dots[0])
                ax.plot(dfGroup[dfGroup > 4*12*7]/dfGroup[dfGroup > 4*12*7]*100,'.', markersize=vaildWavelengthsCount*3-count*2, c=dots[0].get_color()) # Placing entries with more than 100% availability on the 100% line
                print('Dropped ', len(self.__data.loc[StartDate:EndDate, iWaveLength])-len(self.__data.loc[StartDate:EndDate, iWaveLength].dropna()), ' NaN entries from ', iWaveLength)
                count += 1

        ## Formatting the graph
        plt.gcf().autofmt_xdate()
        plt.grid(which='major',axis='both')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7, tz='US/Pacific'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=24, tz='US/Pacific'))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.ylim(0,100)
        plt.ylabel('Availability %')
        plt.legend(handles=handlesList, loc='best')
        plt.tight_layout()

        ## Optional, saving the plot to a file as a .png.
        ### Note: You must save the plot before calling plt.show(), additionally, you must have the relative directory setup otherwise this will produce a "soft" error.
        ### Change 'False' in the if-statement to True to enable saving the plot as a png
        if True:
            filename = self.__siteName + '_' + filename[14:18] + '_' + str(pd.Timestamp.now().strftime('%Y-%m-%d_%H%M%S'))# + StartDate + '_' + EndDate
            location = 'graphs\\availability\\' + filename
            plt.savefig(location) # Saves the plot to a .png file.

        plt.show()

    def _convertDataToTensor(self):
        print(self.__data.empty)
        if not self.__data.empty:
            print(self.__data.head())
            self.__tensorData = convert_to_tensor(self.__data.values, dtype=float64)
            print(self.__tensorData)
# end DataHandler