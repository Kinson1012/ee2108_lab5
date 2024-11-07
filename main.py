import numpy as np
import matplotlib.pyplot as plt

# Define the data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3.1, 5.9, 8.2, 10.5, 12.7, 15.3, 17.5, 19.8, 21.9, 24.2])

# plot the data y against x
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data')
plt.grid(True)
plt.show()

# Calculate the sums
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_squared = np.sum(x ** 2)

# Calculate a1 and a0 using the summation formulas
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
a0 = (sum_y - a1 * sum_x) / n

# Output the values of a0 and a1
print(f"a0: {a0}")
print(f"a1: {a1}")
print()

x_b = np.array([11, 12, 13, 14, 15])
y_b = np.array([26.425, 28.65, 30.875, 33.1, 35.325])
y_pred_b = a0 + a1 * x_b

# print the predicted values
print(f"Predicted values by using linear model: {y_pred_b}\n")

# calculate squared error for each prediction
squared_error = (y_b - y_pred_b) ** 2
print(f"Squared error by using linear model: {squared_error}\n")

# compute the average squared error for all predictions
average_squared_error = np.mean(squared_error)
print(f"Average squared error by using linear model: {average_squared_error}\n")

# fit a polynomial model to data x and y
# determine the a0, a1, a2
X = np.vstack([np.ones(n), x, x ** 2]).T
Y = y[:, np.newaxis]
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
a0, a1, a2 = coefficients.flatten()
print(f"a0: {a0}, a1: {a1}, a2: {a2}\n")

# predict the table c
x_c = np.array([11,12,13,14,15])
y_c = np.array([26.425, 28.65, 30.875, 33.1, 35.325])
y_pred_c = a0 + a1 * x_c + a2 * x_c ** 2
print(f"Predicted values by using polynomial model: {y_pred_c}\n")

# calculate squared error for each prediction
squared_error = (y_c - y_pred_c) ** 2
print(f"Squared error by using polynomial model: {squared_error}\n")

# compute the average squared error for all predictions
average_squared_error = np.mean(squared_error)
print(f"Average squared error by using polynomial model: {average_squared_error}\n")


