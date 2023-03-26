# stock_price_prediction
the code is for predicting the stock prices of a company using an LSTM model.
It uses the Tiingo API to fetch the historical stock prices data of Apple and then performs some data pre-processing and trains an LSTM model to predict the future prices.
to create a api token go to tiingo website login and create your own api token and then paste in code where key is witten 
The model is then used to predict the future prices of the next 30 days.                                     Here is a brief overview of the code:
The required libraries are imported and the API key is pasted.
The historical stock prices data of Apple is fetched using the Tiingo API, and is then saved in a CSV file.
The 'close' column of the data is extracted and scaled using the MinMaxScaler.
The data is then split into training and testing sets, and then transformed into the required format for training the LSTM model.
A stack of three LSTM layers and a dense output layer is created, and the model is compiled.
The model is then trained using the training data and validated using the testing data.
The model is then used to predict the future prices of the next 30 days.
