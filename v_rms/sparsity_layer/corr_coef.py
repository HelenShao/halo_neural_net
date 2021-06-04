import numpy as np

# Calculate R^2 (manually):
def r_squared(input, output):
    # True Vs Predicted Data
    x = input
    y = output

    # Regression line
    # y=x plot
    min = np.min([np.min(input), np.min(output)])
    max = np.max([np.max(input), np.max(output)])
    x_true = np.linspace(min, max, 1000) 
    y_true = x

    # Mean of output values
    mean_line = np.full(len(x),[y.mean()])

    # Squared differences between data points and y=x line
    differences_y = y_true-y
    differences_sum1 = 0
    for i in differences_y:
        differences_sum1 = differences_sum1 + (i**2)
    differences_sum1

    # Squared differences between data points and mean line
    differences_mean = mean_line-y
    differences_sum2 = 0
    for i in differences_mean:
        differences_sum2 = differences_sum2 + (i**2)
    differences_sum2

    # R squared = Explained variance of model / Total variance 
    r_squared = (differences_sum2 - differences_sum1)/differences_sum2
    
    return r_squared
