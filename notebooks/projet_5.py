import pandas as pd 
import matplotlib.pyplot as plt #pour créer et visualiser les données
import numpy as np 

#boston = pd.read_csv('C:/Users/utilisateur/Google Drive/microsoft_ia/Google Drive/projets/projet_5_ia/boston_house_prices.csv', sep=",")
#salaries = pd.read_csv('Position_Salaries.csv', sep=",")
# pinard = pd.read_csv('qualite-vin-rouge.csv', sep=",")
simple = pd.read_csv('C:/Users/utilisateur/Google Drive/microsoft_ia/Google Drive/projets/projet_5_ia/reg_simple.csv', sep=",")

# boston()
# salaries()
# pinard()
print(simple.head())

# on crée les vecteurs x et y
x = np.array((simple['heure_rev'])) # on attribue la fonction x à heure et on crée le vecteur
y = np.array((simple['note'])) # on attribue la fonction Y à note
print(x.shape)
print(y.shape)

x = x.reshape((x.shape[0],1)) # on redimensionne en prenant le nombre de ligne qu'il y a déjà dans x
y = y.reshape((y.shape[0],1))

print(x)
print(y)

plt.scatter(x, y)# afficher les résultats. x en abscisse et y en ordonnée
plt.show() #afficher le nuage de points

# on crée la matrice X
X = np.hstack((x, np.ones(x.shape))) # ca permet de coller ensemble 2 vecteurs l'un à coté de l'autre (il est de même dimension que X) des 1 pour pouvoir faire le calcul matriciel) X-1
print(X.shape)
 
# création d'un vecteur parametre theta (on ne le connait pas c'est la machine qui le calcul et qui nous permet d'avoir l'erreur la plus petite cad minimise la fonction coût)
theta = np.random.randn(2, 1) #(on initialise theta avec des paramêtres aléatoires de dimension (2,1) car notre vecteur contient seulement 2 éléments (a et b) car on fait une régression linéaire f(x) = ax +b

print("theta :" ,theta)
 
# A ce stade on a nos 2 vecteurs (x et y) et nos 2 matrices X et theta

# 2 le modèle linéaire  : on crée une fonction F +X
def model(X, theta):
    return X.dot(theta) # la fonction nous retourne le produit matriciel de X par theta
 # fonction coût
def cost_function(X, y, theta): # on calcule la fonction coût qui est l'erreur quadratique moyenne
    m = len(y) # nombre d'exemple qu'on a dans notre datasale et qui est aussi long que le vecteur y
    return 1/(2*m) * np.sum((model(X, theta) - y)**2) # carré de la différence entre notre modele et y
print(cost_function(X, y, theta))
# fonction calcul du gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)#XT transposé de x
# descente de gradient
def gradient_descent(X, y, theta, learning_rate, n_iterations): #learning rate variable alpha, 
    # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    cost_history = np.zeros(n_iterations) # ceci permet  de créer un tableau rempli de 0 et il est aussi long que nos itérations
     
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) # mise a jour du parametre theta (formule du gradient descent) pendant n itérations
        cost_history[i] = cost_function(X, y, theta) # on enregistre la valeur du Cout au tour i dans cost_history[i]
         
    return theta, cost_history
#machine learning test
n_iterations = 20
learning_rate = 0.001 # factuers sur lequel on peut jouer si notre modèle est trop loin de 1
 
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
 
print("theta_final :" ,theta_final) # voici les parametres du modele une fois que la machine a été entrainée
 
# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, theta_final)
 
# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x, y) # notre dataset
plt.plot(x, predictions, c='r') # r on affiche en rouge
plt.show()

plt.plot(range(n_iterations), cost_history)
plt.show()

#def coefficient R2
# prediction = model(X, theta_final)
# coefficient de determination objectif plus près de 1
rtwo = 1 - ((np.sum(y-predictions)**2)/np.sum(y-np.mean(y)**2))
print("R2 : ", rtwo)