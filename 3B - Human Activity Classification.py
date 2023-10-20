import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy import linalg as LINEARALG
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import Dataset
df = pd.read_excel(r"C:\Users\Bhuvan Karthik\Desktop\INTRO TO ML\Assignment 1 - EECE5644\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\x_train2.xlsx") # encoding='ISO-8859-1')
Data = df.to_numpy()

N = Data.shape[0]          
Y = pd.read_csv(r"C:\Users\Bhuvan Karthik\Desktop\INTRO TO ML\Assignment 1 - EECE5644\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt") #use csv for txt file as well
label = np.squeeze(Y.to_numpy())
print(Data.shape)

# Normalizing data to apply PCA
Data = Data[:, 0:-2]
sc = StandardScaler()  #Mean is 0 and standard deviation is 1 due to this function.
Data = sc.fit_transform(Data)
print("Shape of Data:", Data.shape)

pca = PCA(n_components = 10)   # Reducing dimensions to obtain 10 principal components
Data = pca.fit_transform(Data) #Transforms the data to fit the new model which has lesser features

N_l = 6            # Number of labels 
N_f = 10           # Number of features

# Compute Mean Vectors and Covariance matrices
mean = np.zeros(shape = [N_l, N_f])
Covariance = np.zeros(shape = [N_l, N_f, N_f])

for i in range(0, N_l):
    mean[i, :] = np.mean(Data[(label == i + 1), :], axis = 0)
    Covariance[i, :, :] = np.cov(Data[(label == i + 1), :], rowvar = False)
    Covariance[i, :, :] += (0.00001) * ((np.trace(Covariance[i,:,:]))/LINEARALG.matrix_rank(Covariance[i,:,:])) * np.eye(10)

# Assign 0-1 loss matrix
loss_matrix = np.ones(shape = [N_l, N_l]) - np.eye(N_l) #Creates identity matrix and subtracts it from main one.
P_x_given_L = np.zeros(shape = [N_l, N])
for i in range(0, N_l):
    P_x_given_L[i, :] = multivariate_normal.pdf(Data, mean = mean[i, :], cov = Covariance[i, :,:])
priors = np.zeros(shape = [N_l, 1])
for i in range(0, N_l):
    priors[i] = (np.size(label[np.where((label == i + 1))])) / N  #Priors calculation
P_x = np.matmul(np.transpose(priors), P_x_given_L)
ClassPosteriors = (P_x_given_L * (np.matlib.repmat(priors, 1, N))) / np.matlib.repmat(P_x, N_l, 1)
ExpectedRisk = np.matmul(loss_matrix, ClassPosteriors)
Decision = np.argmin(ExpectedRisk, axis = 0)
print("Minimum Expected Risk", np.sum(np.min(ExpectedRisk, axis = 0)) / N)

# Estimate Confusion Matrix
Confusion = np.zeros(shape = [N_l, N_l])
for d in range(N_l):
    for l in range(N_l):
        Confusion[d, l] = (np.size(np.where((d == Decision) & (l == label - 1)))) / np.size(np.where(label - 1 == l))

np.set_printoptions(suppress=True)
print(Confusion)

# Plot Data Distribution
fig = plt.figure()
ax = plt.axes(projection = "3d")
for i in range(1, N_l + 1):
    ax.scatter(Data[(label==i),1],Data[(label==i),2],Data[(label==i),3], label=i)
plt.xlabel('X3')
plt.ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('Data Distribution')
plt.show()
