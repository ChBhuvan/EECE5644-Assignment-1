import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)
plt.rcParams['figure.figsize'] = [7,7]
from numpy import linalg as LINEARALG

N_f = 4          #Features  
N_S = 10000       #Samples
N_l = 2            #Labels  

# Mean vectors
mean = np.ones(shape=[N_l, N_f])
mean [0, :] = [-1,-1,-1,-1]                      

# Covariance matrices
covariance = np.ones(shape=[N_l, N_f, N_f])            
covariance [0, :, :] = [[2, -0.5, 0.3, 0],[-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
covariance [1, :, :] = [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]
np.random.seed(10)
priors = [0.35, 0.65] 
label = (np.random.rand(N_S) >= priors[1]).astype(int)

# Generate Gaussian distribution for 10000 samples using mean and covariance matrices
X = np.zeros(shape = [N_S, N_f])
for j in range(N_S): 
        if (label[j] == 0):
                X[j, :] = np.random.multivariate_normal(mean[0, :], covariance[0, :,:])
        elif (label[j] == 1):
                X[j, :] = np.random.multivariate_normal(mean[1, :], covariance[1, :,:])

# Compute between-class and within-class scatter matrices
Sb = (mean[0, :] - mean[1, :]) * np.transpose (mean[0, :] - mean[1, :])
Sw = covariance[0, :, :] + covariance[1, :, :]
V, W = LINEARALG.eig(LINEARALG.inv(Sw) * Sb)
W_LDA = W[np.argmax(V)]
X0 = X[np.where(label == 0)]
X1 = X[np.where(label == 1)]

Y0 = np.zeros(len(X0))  # Data projection using wLDA
Y1 = np.zeros(len(X1))
Y0 = np.dot(np.transpose(W_LDA), np.transpose(X0))
Y1 = np.dot(np.transpose(W_LDA), np.transpose(X1))

# Ranging threshold from minimum to maximum
Y = np.concatenate([Y0, Y1])
sort_Y = np.sort(Y)
tau_sweep = []

for i in range(0,9999):
        tau_sweep.append((sort_Y[i] + sort_Y[i+1])/2.0)
decision = []
TP = [None] * len(tau_sweep)
FP = [None] * len(tau_sweep)
minPerror = [None] * len(tau_sweep)

for (index, tau) in enumerate(tau_sweep):
        decision = (Y >= tau)
        TP[index] = (np.size(np.where((decision == 1) & (label == 1))))/np.size(np.where(label == 1))
        FP[index] = (np.size(np.where((decision == 1) & (label == 0))))/np.size(np.where(label == 0))
        minPerror[index] = (priors[0] * FP[index]) + (priors[1] * (1 - TP[index]))
        
gamma_ideal = np.log(priors[0] / priors[1])  #Classify using class priors
ideal_decision = (Y >= gamma_ideal)
TP_ideal = (np.size(np.where((ideal_decision == 1) & (label == 1))))/np.size(np.where(label == 1))
FP_ideal = (np.size(np.where((ideal_decision == 1) & (label == 0))))/np.size(np.where(label == 0))
minPerror_ideal = (priors[0] * FP_ideal) + (priors[1] * (1 - TP_ideal))
print("Gamma Ideal - %f and corresponding minimum error %f" %(np.exp(gamma_ideal), minPerror_ideal))

# Plot ROC curve
plt.plot(FP, TP, color = 'orange')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('Receiver Operating Characteristic(ROC Curve)')
plt.plot(FP[np.argmin(minPerror)], TP[np.argmin(minPerror)],'o',color = 'blue')
plt.show()
print("Gamma Practical - %f and corresponding minimum error %f" %(np.exp(tau_sweep[np.argmin(minPerror)]), np.min(minPerror)))
