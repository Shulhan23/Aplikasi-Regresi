import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

df = pd.read_csv(io.BytesIO(uploaded['Student_Performance (1).csv']))
data = data[['Hours Studied', 'Sample Question Papers Practiced','Performance Index']]
X = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

plt.scatter(X, y, color='yellow', label='Dataset')
plt.plot(X, y_pred_linear, color='red', label='Hasil Regresi Linear')
plt.xlabel('Waktu Belajar')
plt.ylabel('Performa')
plt.title('Regresi Linear')
plt.legend()
plt.show()

rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f'Nilai MSE Regresi Linear: {rms_linear}')

def power_regression(x, a, b):
    return a * np.power(x, b)

params_power, covariance_power = curve_fit(power_regression, X.flatten(), y)
y_pred_pangkat = power_regression(X, *params_power)


plt.scatter(X, y, color='Yellow', label='Dataset')
plt.plot(X, y_pred_pangkat, color='Red', label='Hasil Regresi Pangkat Sederhana')
plt.title('Regresi Pangkat Sederhana')
plt.xlabel('Waktu Belajar (X)')
plt.ylabel('Nilai Ujian (y)')
plt.legend()
plt.show()


rms_power = np.sqrt(mean_squared_error(y, y_pred_pangkat))
print(f'Nilai MSE Regresi Pangkat Sederhana: {rms_power}')
