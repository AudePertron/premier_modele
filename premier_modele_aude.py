import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



np.random.seed(8)
simple = pd.read_csv('C:/Users/utilisateur/Documents/microsoft_ia/premier_model/reg_simple.csv',sep = ",")

print(simple.head())
print(list(simple.columns))
print (simple.heure_rev.dtypes)

#plt.scatter(simple['heure_rev'], simple['note'])
#plt.show()

x = np.array((simple['heure_rev']))
y = np.array((simple['note']))
x = x.reshape((x.shape[0],1))
y = y.reshape((y.shape[0],1))

print("avant ", x.shape)

def uno (x):
    X = np.hstack((x, np.ones(x.shape)))
    return X

X = uno(x)
print("après", X.shape)

theta = np.random.randn(2, 1)

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
print(theta_final)

prediction = model(X, theta_final)

rtwo = 1 - ((np.sum(y-prediction)**2)/np.sum(y-np.mean(y)**2))
print("R2 : ", rtwo)


plt.scatter(x, y)
plt.plot(x, prediction, c='r')
plt.show()

#visualisation courbe de cout
plt.plot(range(n_iterations), cout)
plt.show()


print("********************* version sklearn*********************")

x = np.array((simple['heure_rev']))
y = np.array((simple['note']))
X = x.reshape((x.shape[0],1))
y = y.reshape((y.shape[0],1))

reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
y_pred = reg.predict(X)
r_twosk = r2_score(y, y_pred)
print("R2 sklearn = " + str(r_twosk))
print("il existe une différence entre la méthode manuelle et la méthode automatique, mais elle est minime")
#plot
plt.scatter(X,y)
plt.plot(X, y_pred, c = 'red')
plt.ylabel('note')
plt.xlabel('heure')
plt.title('Evolution')
plt.grid(True)
plt.show()
