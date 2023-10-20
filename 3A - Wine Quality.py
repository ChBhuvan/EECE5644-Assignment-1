import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy import linalg as LINEARALG

# Import Dataset
df = pd.read_excel(r"c:\Users\Bhuvan Karthik\Desktop\INTRO TO ML\Assignment 1 - EECE5644\Wine quality dataset\winequality-white.xlsx")
Data = df.to_numpy()
print(df.columns)
print(df.shape)

N = Data.shape[0]       # Number of Samples
label = Data[:, 11]     # Seperate column containing class labels
Data = Data[:, 0:11]    # Feature set
N_l = 11                # Number of classes
N_f = 11                # Number of features
mean_mat = np.zeros(shape = [N_l, N_f])
covariance = np.zeros(shape = [N_l, N_f, N_f])

# Compute Mean Vectors and Covariance matrices
for i in range(0, N_l):
    mean_mat[i, :] = np.mean(Data[(label == i)], axis = 0)
    if (i not in label):
        covariance[i, :, :] = np.eye(N_f)
    else:
        covariance[i, :, :] = np.cov(Data[(label == i), :], rowvar = False)
        covariance[i, :, :] += (0.000000005) * ((np.trace(covariance[i, :, :]))/LINEARALG.matrix_rank(covariance[i, :, :])) * np.eye(N_f) #lambda is here.

Loss_mat = np.ones(shape = [N_l, N_l]) - np.eye(N_l)
P_x_given_L = np.zeros(shape = [N_l, N])
for i in range(0, N_l):
    if i in label:
        P_x_given_L[i, :] = multivariate_normal.pdf(Data, mean = mean_mat[i, :], cov = covariance[i, :,:])

priors = np.zeros(shape = [11, 1])
for i in range(0, N_l):
    priors[i] = (np.size(label[np.where((label == i))])) / N  #Priors calculation

P_x = np.matmul(np.transpose(priors), P_x_given_L)
ClassPosteriors = (P_x_given_L * (np.matlib.repmat(priors, 1, N))) / np.matlib.repmat(P_x, N_l, 1)
ExpectedRisk = np.matmul(Loss_mat, ClassPosteriors)
Decision = np.argmin(ExpectedRisk, axis = 0)
print("Minimum Expected Risk", np.sum(np.min(ExpectedRisk, axis = 0)) / N)

# Estimate Confusion Matrix
Confusion = np.zeros(shape = [N_l, N_l])
for d in range(N_l):
    for l in range(N_l):
        if l in label and d in label:
            Confusion[d, l] = (np.size(np.where((d == Decision) & (l == label)))) / np.size(np.where(label == l))
print(np.array2string(np.round(Confusion, 5), separator=','))  #print(Confusion)

# Plot Data Distribution
fig = plt.figure()
ax = plt.axes(projection = "3d")
for i in range(N_l):
    ax.scatter(Data[(label==i),1],Data[(label==i),2],Data[(label==i),3], label=i)
plt.xlabel('X3')
plt.ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('Data Distribution')
plt.show()
