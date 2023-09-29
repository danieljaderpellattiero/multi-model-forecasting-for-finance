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
- The models will locally save the stocks data, in order to allow multiple models to be built from them. ( _data will be located inside the `./M{model_number}/data` folder_ )
- The models will locally save the deep learning models generated for each stock, so that they can be reloaded directly from the disk. ( _models will be located inside the `./M{model_number}/models` folder_ )
- The models will locally export the graphs of each stocks datasets and the graphs of the respective forecasts. ( _graphs will be located inside the `./M{model_number}/images/data-preprocessing` and `./M{model_number}/images/predictions` folders_ )
- The models will export in a csv file the forecasted values and all the error metrics needed to use the predictions of the various models jointly. ( _csv files will be located inside the `./M{model_number}/predictions` folder_ )

> NdR: models code documentation will be enriched at a later date.

---

## The Multi-model

### How to use it

> Assumption: suppose we ran the models for 1 test run on the MSFT ticker.

1. Create inside `./MultiModel/data` a folder named `MSFT`.
2. Place inside the aforementioned folder the stock data. ( _data can be taken from `./M3/data/MSFT/test_run_0/[training|validation|test].csv`\, taking care to add the "test\_run\_0\_" prefix to each csv file name_ )
3. Create inside `./MultiModel/models_predictions/M{model_number}` a folder named `MSFT` and place inside it ( them ) the respective predictions of each model. ( _predictions can be taken from the `./M{model_number}/predictions/MSFT/` folder_ )
4. Run the Multi-model by launching the `./MultiModel/Main.py` file.
5. The Multi-model will export the graphs for each stocks single and ensembled predictions. ( _graphs will be located inside the `./MultiModel/images/MSFT/single_predictions` and `./MultiModel/images/MSFT/ensembled_predictions` folders_ )
6. The Multi-model will export in a csv file the metrics for each stocks single and ensembled predictions. ( _csv files will be located inside the `./MultiModel/results/MSFT` folder_ )

> NdR: I address the term "single predictions" to the ones generated from the single models, whereas the "ensembled" are the ones generated by jointly using the models. ( _the way the models predictions are combined is described thoroughly in the thesis_ )
