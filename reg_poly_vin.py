import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

vin = pd.read_csv('C:/Users/utilisateur/Documents/GitHub/premier_modele/Data_Regression/qualite-vin-rouge.csv',sep = ",")

#récupération de X et y
X = vin.iloc[:,:-1].values
y = vin.iloc[:,-1].values
print(X.shape)
y = y.reshape((y.shape[0],1))
print(y.shape)

X2 = np.square(X)
#X3 = np.power(X, 3) -->overflow

X = np.c_[X, X2] 

print(X.shape)

#ajout de la colonne de 1 à X
def un (x):
    m = x.shape[0]
    n = x.shape[1]
    x = np.c_[ np.ones(m), x ] 
    print(x.shape)
    return x

X = un(X)

n = X.shape[1]

theta = np.random.randn(n, 1)

def model(X, theta):
    return X.dot(theta)

def fonction_cout(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
 
def gradient(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def descente_gradient(X, y, theta, alpha, n_iterations):
    cout = np.zeros(n_iterations) 
     
    for i in range(0, n_iterations):
        theta = theta - alpha * gradient(X, y, theta)
        cout[i] = fonction_cout(X, y, theta)
         
    return theta, cout

n_iterations = 20
alpha = 0.001

theta_final, cout = descente_gradient(X, y, theta, alpha, n_iterations)

prediction = model(X, theta_final)

mse = mean_squared_error(y, prediction)
print("mean squared error :" + str(mse) )


#visualisation courbe de cout
plt.plot(range(n_iterations), cout)

plt.show()


print("*****************************version sklearn**********************")
X_or = vin.iloc[:,:-1].values
scaler = StandardScaler()
scaler.fit(X_or)
X = scaler.transform(X_or)

polynomial_features = PolynomialFeatures(degree = 4)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)
plt.plot(X_or[:,10], y_pred)

#plt.scatter(X[:,0],y, c='r')
plt.title("regression polynomiale")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

mse_sk = mean_squared_error(y, y_pred)
print("mean squared error polynomiale sklearn: " + str(mse_sk) )