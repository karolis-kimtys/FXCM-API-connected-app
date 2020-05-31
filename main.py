import fxcmpy
import os
import time
#import datetime as dt
import numpy as np
import pandas as pd
#import matplotlib
#from datetime import datetime
#from statistics import mean
#from sklearn.linear_model import LinearRegression
from mplfinance.original_flavor import candlestick2_ohlc
from numpy.polynomial.polynomial import polyfit
from scipy.signal import argrelextrema
#import numpy.ma as ma
#import mplfinance as mpf
import matplotlib.pyplot as plt

err_allowed = 10.0 / 100        #Tolerance
sr_zones = 1                    #Number of points before and after

def func_list():
    print("\nFunctions avaliable:\n")
    
    fun_list = ["connect()",    "disconnect()",     "instruments()",
                "data()",       "clean()",          "peaks()",
                "ema()",        "data_list()",      "best_fit()"]
    x = 0
    for i in fun_list:
        fun = x, i
        x += 1
        print(fun)

    func = input("\nEnter number for function : ")
    func = eval(fun_list[int(func)])
    print("")

def data_list():
    global file
    print("\nFiles avaliable:\n")
    
    file_list = os.listdir("data/")
    
    x = 0
    for i in file_list:
        file = x, i
        x += 1
        print(file)

    file = input("\nEnter number for function : ")
    file = file_list[int(file)]
    file = file.strip('"')

def connect():
    start_time = time.time()
    print("\nConnecting...")
    global con
    con = fxcmpy.fxcmpy(config_file='config.cfg', server='demo')
    if con.is_connected() == True:
        counter = time.time() - start_time
        counter = round(counter)
        print("\nConnected - " + str(counter) + "s\n")
    func_list()

def disconnect():
    con.close()
    print("\nDisconnected")
    func_list()

def instruments():
    print("\nAvailable instrument list:")
    instrument = con.get_instruments()
    print(instrument)
    func_list()
    print("")

def data():
    inst = input("Enter name of the instrument 'str':")
    perd = input("Enter timeframe 'str':")
    numb = input("Enter number of points 'int':")
    data = con.get_candles(inst, period = perd, number = int(numb))
    print("Head - START")
    print(data.head())
    print("Tail - END")
    print(data.tail())
    print("Enter name for CSV File")
    filename = input("Enter name :")
    data.to_csv("data/" + filename + "_" + perd + ".csv")
    print(filename + " data saved to .csv")
    func_list()

def clean():
    data_list()
    df = pd.read_csv("data/" + file)
    df["Date"] = df["date"]
    df.set_index("Date", inplace=True)
    df["Open"] = (df["bidopen"] + df["askopen"]) / 2
    df["High"] = (df["bidhigh"] + df["askhigh"]) / 2
    df["Low"] = (df["bidlow"] + df["asklow"]) / 2
    df["Close"] = (df["bidclose"] + df["askclose"]) / 2
    df["Volume"] = df["tickqty"]
    df.drop(columns=["date", "bidopen", "askopen", "bidhigh", "askhigh",\
                     "bidlow", "asklow", "bidclose", "askclose", "tickqty"],\
                     axis = 1, inplace=True)
    df.to_csv("data/" + file)
    print(df)
    print(os.listdir("data/"))
    print("\nCSV Cleaned")
    func_list()

def peaks():
    data_list()
    global df
    df = pd.read_csv("data/" + file)
    df.set_index("Date", inplace=True)
    df['Support'] = df.iloc[argrelextrema\
                    (df["Low"].values, np.less_equal, order=sr_zones)\
                            [0]]["Low"]
    df['Resistance'] = df.iloc[argrelextrema\
                    (df["High"].values, np.greater_equal, order=sr_zones)\
                            [0]]["High"]
    print(df['Support'])
    print(df['Resistance'])
    
    fig, ax = plt.subplots(figsize=[15, 9])
    candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'],\
                      colorup='green', colordown='red', width=0.5)

    plt.scatter(df["Support"].index, df["Support"].values)
    plt.scatter(df["Resistance"].index, df["Resistance"].values)
    plt.show()

    df.to_csv("data/" + file)
    print("\nPeaks detected and Stored in CSV")
    func_list()

def ema():
    data_list()
    global df
    df = pd.read_csv("data/" + file)
    df.set_index("Date", inplace=True)
    ema = pd.Series.ewm(df['Close'], span = 200).mean()
    df['EMA'] = ema
    print(df['EMA'])

    fig, ax = plt.subplots(figsize=[15, 9])
    candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'],\
                   colorup='green', colordown='red', width=0.5)

    plt.plot(ema)
    plt.scatter(df["Support"].index, df["Support"].values)
    plt.scatter(df["Resistance"].index, df["Resistance"].values)
    plt.show()

    df.to_csv("data/" + file)
    print("\nEMA detected and Stored in CSV")
    func_list()

def best_fit():
    start_time = time.time()
    global df
    data_list()
    df = pd.read_csv("data/" + file)
    #df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y %H:%M')

    df["Sup_linear"] = df["Support"]
    x = df["Sup_linear"].index
    y = df["Sup_linear"].to_numpy()

    linear = df["Support"].dropna().tolist()

    print(y)

    

    #mask = np.isnan(y)
    #y[mask] = np.interp(np.flatnonzero(mask),\
                           #np.flatnonzero(~mask), y[~mask])
    
    y_chunk = []

    print(y_chunk)

    for i in range(0, len(linear)-3):
        y_chunk = linear[i+1], linear[i+2], linear[i+3]
        x_chunk = x[i+1], x[i+2], x[i+3]

        
        print(y_chunk)
        print(x_chunk)
        def best_fit(x_chunk, y_chunk):

            xbar = sum(x_chunk)/len(x_chunk)
            ybar = sum(y_chunk)/len(y_chunk)
            n = len(x_chunk) # or len(Y)

            numer = sum([xi*yi for xi,yi in zip(x_chunk, y_chunk)])\
                                            - n * xbar * ybar
            denum = sum([xi**2 for xi in x_chunk]) - n * xbar**2

            b = numer / denum
            a = ybar - b * xbar

            print("best fit line:\ny = ", a, " + ", b, "x")
            

            return a, b

        # solution
        a, b = best_fit(x_chunk, y_chunk)
        #best fit line:
        #y = 0.80 + 0.92x

        # plot points and fit line
        
        plt.scatter(x_chunk, y_chunk)
        yfit = [a + b * xi for xi in x_chunk]

        print("yfit ", yfit)
        plt.plot(x_chunk, yfit)
        #plt.show(block = True)
    counter = time.time() - start_time
    #counter = round(counter)
    print("\n" + str(counter) + "s\n")



# iterate over linear list with values and find first value and store
  #      in value variable
   #     then         maybe compare y numpy array
    #    if np.array(y) != linear value
     #       then use first value stored in value variable
      #      else if np.array == value
       #     then change value and store in variable
        
 

        

        


    
    # Fit with polyfit
    b, m = polyfit(x, y, 1)


    fig, ax = plt.subplots(figsize=[15, 9])
    candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'],\
                          colorup='green', colordown='red', width=0.5)

    plt.scatter(df["Support"].index, df["Support"].values)
    plt.scatter(df["Resistance"].index, df["Resistance"].values)

    plt.plot(x, y, '.')
    plt.plot(x, b + m * x, '-')

    plt.tight_layout()
    plt.get_current_fig_manager().window.wm_geometry("+200+50")
    plt.show(block = True)

    """
    #df.to_csv("data/" + file)
    print("\nLinear Regression S & R detected and Stored in CSV")
    """
    func_list()

##############################################################################
##############################################################################
##############################################################################

if __name__ == "__main__":
    print("")
    print("--- System has started ---")
    print("")

    func_list()


    pass
