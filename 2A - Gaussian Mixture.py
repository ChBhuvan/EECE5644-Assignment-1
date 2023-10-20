import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)
N = 10000               # Number of Samples
N_l = 3                 # Number of classes
N_f = 3                 # Number of features
N_m = 4                 # Number of gaussian distributions
np.random.seed(10) 
priors = np.array([[0.3, 0.3, 0.4]])

# Covariance matrices
covariance = np.zeros(shape=[N_m, N_f, N_f])
covariance[0, :, :] = 36 * np.linalg.matrix_power((np.eye(N_f)) + (0.01 * np.random.randn(N_f, N_f)), 2)
covariance[1, :, :] = 36 * np.linalg.matrix_power((np.eye(N_f)) + (0.02 * np.random.randn(N_f, N_f)), 2)
covariance[2, :, :] = 36 * np.linalg.matrix_power((np.eye(N_f)) + (0.03 * np.random.randn(N_f, N_f)), 2)
covariance[3, :, :] = 36 * np.linalg.matrix_power((np.eye(N_f)) + (0.04 * np.random.randn(N_f, N_f)), 2)

# Mean vectors
mean = np.zeros(shape=[N_m, N_f])
mean [0, :] = [0, 0, 10]
mean [1, :] = [0, 10, 0]
mean [2, :] = [10, 0, 0]
mean [3, :] = [10, 0, 10]

prior_gmm_label3 = [0.5,0.5]
cumsum = np.cumsum(priors)
randomlabels = np.random.rand(N)
label = np.zeros(shape = [10000])
for i in range(0,N-1):
    if randomlabels[i] <= cumsum[0]:
        label[i] = 0
    elif randomlabels[i] <= cumsum[1]:
        label[i] = 1
    else:
        label[i] = 2 

# Generate gaussian distribution for 10000 samples using mean and covariance matrices
X = np.zeros(shape = [N, N_f])
for i in range(N):
    if (label[i] == 0):
        X[i, :] = np.random.multivariate_normal(mean[0, :], covariance[0, :, :])
    elif (label[i] == 1):
        X[i, :] = np.random.multivariate_normal(mean[1, :], covariance[1, :, :])
    elif (label[i] == 2):
        if (np.random.rand(1,1) >= prior_gmm_label3[1]):
            X[i, :] = np.random.multivariate_normal(mean[2, :], covariance[2, :, :])
        else:
            X[i, :] = np.random.multivariate_normal(mean[3, :], covariance[3, :, :])

loss_matrix = np.ones(shape = [N_l, N_l]) - np.eye(N_l)
#loss_matrix = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]]) -- Loss matrices for 2B 
#loss_matrix = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])
print(loss_matrix)

P_x_given_L = np.zeros(shape = [N_l, N])
for i in range(N_l):
    P_x_given_L[i, :] = multivariate_normal.pdf(X,mean = mean[i, :], cov = covariance[i, :,:])

P_x = np.matmul(priors, P_x_given_L)
ClassPosteriors = (P_x_given_L * (np.matlib.repmat(np.transpose(priors), 1, N))) / np.matlib.repmat(P_x, N_l, 1)
ExpectedRisk = np.matmul(loss_matrix, ClassPosteriors)
Decision = np.argmin(ExpectedRisk, axis = 0)
print("Minimum Expected Risk", np.sum(np.min(ExpectedRisk, axis = 0)) / N)
Confusion = np.zeros(shape = [N_l, N_l])
for d in range(N_l):
    for l in range(N_l):
        Confusion[d, l] = (np.size(np.where((d == Decision) & (l == label)))) / np.size(np.where(label==l))
print(Confusion)

# Plot Classification results
fig = plt.figure()
ax = plt.axes(projection = "3d")
#Use only red and green colors
ax.scatter(X[(label==2) & (Decision == 1),0],X[(label==2) & (Decision == 1),1],X[(label==2) & (Decision == 1),2],color ='red', marker = 'o')
ax.scatter(X[(label==2) & (Decision == 2),0],X[(label==2) & (Decision == 2),1],X[(label==2) & (Decision == 2),2],color ='green', marker = 'o')
ax.scatter(X[(label==2) & (Decision == 0),0],X[(label==2) & (Decision == 0),1],X[(label==2) & (Decision == 0),2],color ='red', marker = 'o')
ax.scatter(X[(label==1) & (Decision == 1),0],X[(label==1) & (Decision == 1),1],X[(label==1) & (Decision == 1),2],color ='green', marker = 's')
ax.scatter(X[(label==1) & (Decision == 2),0],X[(label==1) & (Decision == 2),1],X[(label==1) & (Decision == 2),2],color ='red', marker = 's')
ax.scatter(X[(label==1) & (Decision == 0),0],X[(label==1) & (Decision == 0),1],X[(label==1) & (Decision == 0),2],color ='red', marker = 's')
ax.scatter(X[(label==0) & (Decision == 0),0],X[(label==0) & (Decision == 0),1],X[(label==0) & (Decision == 0),2],color ='green', marker = '^')
ax.scatter(X[(label==0) & (Decision == 2),0],X[(label==0) & (Decision == 2),1],X[(label==0) & (Decision == 2),2],color ='red', marker = '^')
ax.scatter(X[(label==0) & (Decision == 1),0],X[(label==0) & (Decision == 1),1],X[(label==0) & (Decision == 1),2],color ='red', marker = '^')
plt.xlabel('X1')
plt.ylabel('X2')
ax.set_zlabel('X3')
plt.title('Classification Plot')
plt.show()
