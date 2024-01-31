from model import Prediction

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def make_prediction(inputs: list[float], outputs: list[float], input_value: float, plot:bool = False) -> Prediction:
    if len(inputs) != len(outputs):
        raise Exception('Length of "inputs" and "outputs" must match.')

    # Create a dataframe for our data
    df = pd.DataFrame({'inputs':inputs, 'outputs':outputs})

    # Reshape the data using Numpy (x: Inputs, y: Outputs)
    x = np.array(df['inputs']).reshape(-1, 1)
    y = np.array(df['outputs']).reshape(-1, 1)

    # Split the data into training data to test our model
    train_x, test_x, train_y, test_y = train_test_split(x,y,random_state=0, test_size=.20)
    
    # Initialize the model and test it
    model = LinearRegression()
    model.fit(train_x, train_y)

    # Prediction
    y_prediction = model.predict([[input_value]])
    y_line = model.predict(x)

    # Testing for accuracy
    y_test_prediction = model.predict(test_x)

    # Plot
    if plot:
        raise NotImplementedError('Function not there yet')

    return Prediction(value=y_prediction[0][0],
                      r2_score=r2_score(test_y,y_test_prediction),
                      slope=model.coef_[0][0],
                      intercept=model.intercept_[0],
                      mean_absolute_error=mean_absolute_error(test_y,y_test_prediction))


make_prediction([1,2],[3,4], 0)