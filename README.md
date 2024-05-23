# Predicting Future Aerosol Optical Depth Values with Aeronet Data
## A project for Neural Networks (CS-3400/COGS-4680) at CSU Stanislaus

Data can be found [here](https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3).

The final report for this project can be found [here](https://github.com/CastilloAnthony/Neural_Nets/blob/main/Neural%20Nets%20Final%20Project%20Report.pdf).

Explanation of primary files in the repository:
- The dataset contains air quality data collected from the publicly available NASA Aerosol Robotic Network, termed Aeronet. It includes multiple atmospheric aerosol properties using a spectral radiometer that measures Sun and sky radiances at a number of fixed wavelengths within the visible and near-infrared spectrum. 
- The ResidualWrapper class is a custom TensorFlow Keras model wrapper that implements a residual connection. Residual connections are often used in deep learning models to help mitigate the vanishing gradient problem by allowing gradients to flow through the network more easily.
- The DataHandler class is made to perform preprocessing functions on the data, such as data normalization and the integration of a time of day signal. The class also integrates the data into concise time-series windows, and coordinates the flow of data between the windows and the model itself.
- The WindowGenerator class is made to handle the preparation and processing of time series data for model training. It creates input-output pairs from the raw time series data, which are then used for training, validation, and testing.
- The Arbiter class is responsible for managing the overall workflow of data preparation, model training, and evaluation. It coordinates the different components such as data loading, window generation, and model training.
- The Main class is responsible for starting and running the program, it controls how and when the functions of the Arbiter class gets called.
