# Multi-model forecasting for finance

## Learning objectives and skills to be acquired

The student will acquire skills in the design, implementation and validation of machine learning pipelines, with specific reference to the use of algorithms and multiple models for the prediction of financial indicators.  
From a software perspective, the student will acquire skills in the use of Python and major Machine Learning libraries, especially algorithms capable of working on time-series data.

## Organization of the internship

The internship includes an initial phase of state-of-the-art analysis and identification of both the most relevant Machine Learning approaches and the most interesting and challenging tasks in the world of financial trading.  
This will be followed by a critical evaluation of what has been learned in order to identify a task of specific interest, along with a set of the most promising algorithms.
  
Finally, a software that can effectively and efficiently combine these algorithms in order to improve their overall performance will be implemented.

---

## Machine Learning models

### WSAEs-LSTM ( model №1 )

#### How to use it

1. Open the project folder using a Python IDE. ( _preferably [PyCharm](https://www.jetbrains.com/pycharm/)_ )
2. Install the dependencies.
3. Open the file `Main.py` and insert inside the second constructor parameter all the desired stocks' tickers as strings. ( _just like the pre-inserted ticker "MSFT"_ )
4. Launch the model using the same file.

#### Further notes about the model

- The model will print out step-by-step notifications in the terminal from which it was launched.
- The model will locally save the stocks data, in order to allow multiple models to be built from them. ( _data will be located inside the `./data` folder_ )
- The model will locally export the graphs of each stock's datasets and the graphs of the respective price trend forecasts. ( _graphs will be located inside the `./images/data-preprocessing` and `./images/predictions` folders_ )
- The model will locally save the ML models generated for each stock, so that they can be reloaded directly from the disk. ( _models will be located inside the `./models` folder_ )

> NdR:
>
> - Model code documentation will be enriched at a later date.
