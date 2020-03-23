import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# importing data
df = pd.read_csv("Logistic_data.csv")
df.columns = ['Exam1', 'Exam2', 'Admitted']
# training the model
X = df[['Exam1', 'Exam2']]
Y = df['Admitted']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print("Accuracy = " + str(log_reg.score(X_test, y_test)))
# plotting data
plt.title("Admittance Decision maker")
plt.xlabel("Exam1")
plt.ylabel("Exam2")
for n, i in enumerate(df['Admitted']):
    if i == 1:
        plt.scatter(df['Exam1'][n], df['Exam2'][n], color='b', marker='o')
    elif i == 0:
        plt.scatter(df['Exam1'][n], df['Exam2'][n], color='r', marker='x')
points_x=[x for x in range(20,+100)]
line_bias = log_reg.intercept_
line_w = log_reg.coef_.T
points_y=[(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in points_x]
plt.plot(points_x, points_y)
plt.show()
