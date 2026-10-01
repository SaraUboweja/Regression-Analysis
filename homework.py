import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
print(df.head())
X = df[['X']]
y = df['Y']
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
print("Regression Equation:")
print("Y =", slope, "X +", intercept)
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()