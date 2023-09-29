# Multi-model forecasting for finance

## Learning objectives and skills to be acquired

The student will acquire skills in the design, implementation and validation of machine learning pipelines, with specific reference to the use of algorithms and multiple models for the prediction of financial indicators.  
From a software perspective, the student will acquire skills in the use of Python and major Machine Learning libraries, especially algorithms capable of working on time-series data.

## Organization of the internship

The internship includes an initial phase of state-of-the-art analysis and identification of both the most relevant Machine Learning approaches and the most interesting and challenging tasks in the world of financial trading.  
This will be followed by a critical evaluation of what has been learned in order to identify a task of specific interest, along with a set of the most promising algorithms.
  
Finally, a software that can effectively and efficiently combine these algorithms in order to improve their overall performance will be implemented.

---

## Deep Learning models

### WSAEs-LSTM ( model №1 )

#### Paper

- [Recurrent Neural Networks for Financial Time-Series modelling](https://ieeexplore.ieee.org/abstract/document/8545666)

### CEEMDAN-LSTM ( model №2 )

#### Papers

- [Financial time series forecasting model based on CEEMDAN and LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0378437118314985)

- [Carbon price forecasting based on CEEMDAN and LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0306261922000782?via=ihub)

### CNN-LSTM ( model №3 )

#### Paper

- [A CNN–LSTM model for gold price time-series forecasting](https://link.springer.com/article/10.1007/s00521-020-04867-x)

### How to use them

1. Open the project folder using a Python IDE. ( _preferably [PyCharm](https://www.jetbrains.com/pycharm/)_ )
2. Install the required dependencies.
3. Open the `Main.py` file and insert, inside the second constructor parameter, all the desired stocks' tickers as strings. ( _just like the pre-inserted ticker "MSFT"_ )
4. Launch the model using the same file.

### Further notes about the models

- The models will print out step-by-step notifications in the terminal from which they are launched.
- The models will locally save the stocks data, in order to allow multiple models to be built from them. ( _data will be located inside the `./data` folder_ )
- The models will locally save the deep learning models generated for each stock, so that they can be reloaded directly from the disk. ( _models will be located inside the `./models` folder_ )
- The models will locally export the graphs of each stock's datasets and the graphs of the respective forecasts. ( _graphs will be located inside the `./images/data-preprocessing` and `./images/predictions` folders_ )
- The models will export in a csv file the forecasted values and all the error metrics needed to use the predictions of the various models jointly. ( _csv files will be located inside the `./predictions` folder_ )

> NdR: Models code documentation will be enriched at a later date.
