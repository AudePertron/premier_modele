import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


np.random.seed(8)

boston = pd.read_csv('C:/Users/utilisateur/Documents/GitHub/premier_modele/Data_Regression/boston_house_prices.csv',sep = ",")


print(boston.head())
print(list(boston.columns))

#récupération de X et y
X = boston.iloc[:,:-1].values
y = boston.iloc[:,-1].values
print(X.shape)
y = y.reshape((y.shape[0],1))
print(y.shape)

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
alpha = 0.000001

theta_final, cout = descente_gradient(X, y, theta, alpha, n_iterations)

prediction = model(X, theta_final)

mse = mean_squared_error(y, prediction)
print("mean squared error :" + str(mse) )

plt.plot(range(n_iterations), cout)
plt.show()

print("********************* version sklearn*********************")

X = boston.iloc[:,:-1].values
y = boston.iloc[:,-1:].values
print(X.shape)
print(y.shape)

reg = LinearRegression().fit(X, y)

y_pred = reg.predict(X)
print(reg.score(X, y_pred))
mse_sk = mean_squared_error(y, y_pred)

print("mean squared error sklearn: " + str(mse_sk) )


print("*****************************essai avec SGD classifier*******************")

X = boston.iloc[:,:-1].values
y = boston.iloc[:,-1:].values
print(X.shape)
print(y.shape)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
sgd = SGDRegressor(max_iter=1000, learning_rate='adaptive').fit(X,y)
y_sgd = sgd.predict(X)
print("sgd score " + str(sgd.score(X, y_sgd)))


#clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, learning_rate='optimal',))
#clf.fit(X, y)
#Pipeline(steps=[('standardscaler', StandardScaler()),('sgdclassifier', SGDClassifier())])
#y_predSGD = clf.predict(X)
mse_SGD = mean_squared_error(y, y_sgd)
print("mean squared error sklearn: " + str(mse_SGD) )

print("*****************************conclusion*******************")
print("la différence entre la linear regression et le sgd regressor est négligeable sur ce modèle. Par contre elle est immense avec la méthode manuelle, il y a probablement une erreur dedans...")
