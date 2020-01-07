import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# importing data
df = pd.read_csv("housing2.csv")
df.columns = ['Area', 'Num_of_rooms', 'Price']
# feature scaling
df['Area'] = df['Area']/df['Area'].max()
df['Num_of_rooms'] = df['Num_of_rooms']/df['Num_of_rooms'].max()
df['Price'] = df['Price']/df['Price'].max()
area = df['Area']
num_of_rooms = df[['Num_of_rooms']]
price = df['Price']
# training the model
xs = df[['Area', 'Num_of_rooms']]
linear_reg = LinearRegression()
linear_reg.fit(xs, df['Price'])
# defining hypothesis


def hyp(x, y):
    return linear_reg.intercept_ + x*linear_reg.coef_[0] + y*linear_reg.coef_[1]


n = np.linspace(0, 1, 30)
i = np.linspace(0, 1, 30)

X, Y = np.meshgrid(n, i)
Z = hyp(X, Y)
# plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(area, num_of_rooms, price, color='r', marker='o')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
title = 'Hypothesis with cost = ' + str(mean_squared_error(price, linear_reg.predict(xs)))
ax.set_title(title)
ax.set_xlabel("Area")
ax.set_ylabel("Rooms number")
ax.set_zlabel("Price")
plt.show()

