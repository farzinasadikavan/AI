import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
import time
# 111111111111111111111111111111111111111111111111
cars = pd.read_csv('FuelConsumptionCo2.csv')
print(cars.head())
print(cars.tail())
print(cars.describe(include='all'))
# 222222222222222222222222222222222222222222222222
print(cars.info())
cleanup_type = {"FUELTYPE": {"Z": 0, "D": 1, "E": 2, "X": 3}}
cars = cars.replace(cleanup_type)
# 333333333333333333333333333333333333333333333333
enginesize_nan = cars['ENGINESIZE'].isna().sum()
print('Count of ENGINESIZE NaN:'+str(enginesize_nan))
cylinders_nan = cars['CYLINDERS'].isna().sum()
print('Count of CYLINDERS NaN:'+str(cylinders_nan))
fueltype_nan = cars['FUELTYPE'].isna().sum()
print('Count of FUELTYPE NaN:'+str(fueltype_nan))
co2missios_nan = cars['CO2EMISSIONS'].isna().sum()
print('Count of CO2EMISSIONS NaN:'+str(co2missios_nan))
mean_enginsize = cars['ENGINESIZE'].mean()
cars['ENGINESIZE'].fillna(value=mean_enginsize, inplace=True)
mean_cylinders = cars['CYLINDERS'].mean()
cars['CYLINDERS'].fillna(value=mean_cylinders, inplace=True)
mean_fueltype = cars['FUELTYPE'].mean()
cars['FUELTYPE'].fillna(value=mean_fueltype, inplace=True)
new_cars = cars[cars['CO2EMISSIONS'].isna()]
cars.dropna(subset=['CO2EMISSIONS'])
cars = cars[cars['CO2EMISSIONS'].notna()]
cars = cars.reset_index()
# 444444444444444444444444444444444444444444444444
start1 = time.time()
CarsLower240 = cars[cars['CO2EMISSIONS'] < 240]
mean_cars_lower_240 = CarsLower240['FUELCONSUMPTION_CITY'].mean()
print("avrage FUELCONSUMPTION_CITY for co2<240="+str(mean_cars_lower_240))
CarsMoreThan300 = cars[cars['CO2EMISSIONS'] > 300]
mean_cars_more_than_300 = CarsMoreThan300['FUELCONSUMPTION_CITY'].mean()
print("avrage FUELCONSUMPTION_CITY for co2>300="+str(mean_cars_more_than_300))
stop1 = time.time()
# 5555555555555555555555555555555555555555555555555
start2 = time.time()
number = len(cars)
sum = 0
n = 0
for i in range(number):
    if cars['CO2EMISSIONS'][i] < 240:
        n += 1
        sum += cars['FUELCONSUMPTION_CITY'][i]
mean_cars_lower_240 = sum/n
for i in range(number):
    if cars['CO2EMISSIONS'][i] > 300:
        n += 1
        sum += cars['FUELCONSUMPTION_CITY'][i]
mean_cars_more_than_300 = sum/n
stop2 = time.time()
print("with For:avrage FUELCONSUMPTION_CITY for co2<240="+str(mean_cars_lower_240))
print("with For:avrage FUELCONSUMPTION_CITY for co2>300=" +
      str(mean_cars_more_than_300))
print('time vectorization='+str(stop1-start1))
print('time For='+str(stop2-start2))
# 6666666666666666666666666666666666666666666666666
hist = cars.hist(bins=25, figsize=(30, 20))
# 7777777777777777777777777777777777777777777777777
cars_num = cars.select_dtypes(include=[np.number])
cars_normal = (cars_num-cars_num.mean())/cars_num.std()
# 8888888888888888888888888888888888888888888888888
chart1 = cars_normal.plot.scatter(
    x='ENGINESIZE', y='CO2EMISSIONS', title="Scatter plot between ENGINESIZE & CO2EMISSIONS")
chart2 = cars_normal.plot.scatter(
    x='CYLINDERS', y='CO2EMISSIONS', title="Scatter plot between CYLINDERS & CO2EMISSIONS")
chart3 = cars_normal.plot.scatter(x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS',
                                  title="Scatter plot between vFUELCONSUMPTION_CITY & CO2EMISSIONS")
chart4 = cars_normal.plot.scatter(x='FUELCONSUMPTION_HWY', y='CO2EMISSIONS',
                                  title="Scatter plot between FUELCONSUMPTION_HWY & CO2EMISSIONS")
chart5 = cars_normal.plot.scatter(x='FUELCONSUMPTION_COMB', y='CO2EMISSIONS',
                                  title="Scatter plot between FUELCONSUMPTION_COMB & CO2EMISSIONS")
chart6 = cars_normal.plot.scatter(x='FUELCONSUMPTION_COMB_MPG', y='CO2EMISSIONS',
                                  title="Scatter plot between FUELCONSUMPTION_COMB_MPG & CO2EMISSIONS")
# 9999999999999999999999999999999999999999999999999
linear = cars_normal[['FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
X = linear['FUELCONSUMPTION_COMB']
Y = linear['CO2EMISSIONS']
X_mat = np.vstack((np.ones(len(X)), X)).T
# Beta_hat=((xT*X)**-1)*xT*y
beta_hat = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y)
y_hat = X_mat.dot(beta_hat)
mp.scatter(X, Y)
mp.plot(X, y_hat, color='red')
# 10101010101010101010101010101010101010101010
MSE = np.square(np.subtract(Y, y_hat)).mean()
# 1111111111111111111111111111111111111111111111
mp.scatter(X, Y)
mp.plot(X, y_hat, color='red')
# 1212121221212122112211221212121211221211221121
new_cars_num = new_cars.select_dtypes(include=[np.number])
new_cars_normal = (new_cars_num-new_cars_num.mean())/new_cars_num.std()
new_linear = new_cars_normal[['FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
new_x = new_linear['FUELCONSUMPTION_COMB']
new_y = new_linear['CO2EMISSIONS']
new_x_mat = np.vstack((np.ones(len(new_x)), new_x)).T
new_y_hat = new_x_mat.dot(beta_hat)
new_linear = new_linear.reset_index()
new_linear['CO2EMISSIONS'] = new_y_hat
new_linear.to_csv('1.csv')
