import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

vin = pd.read_csv('C:/Users/utilisateur/Documents/microsoft_ia/premier_model/qualite-vin-rouge.csv',sep = ",")

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

sns.pairplot(data=vin, hue="qualité")

plt.show()
