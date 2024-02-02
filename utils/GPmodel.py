# Copyright (c) by Zewen Yang under GPL-3.0 license
# Last modified: Zewen Yang 02/2024

import numpy as np
import numpy.matlib
from ensure import ensure_annotations
from typing import List, Tuple

class GPmodel():
    def __init__(self, x_dim, y_dim, indivDataThersh, 
                 sigmaN, sigmaF, sigmaL, 
                 priorFunc):
        self.DataQuantity = 0
         # user defined values
        self.indivDataThersh = indivDataThersh  # data limit per local GP
        self.x_dim = x_dim  # dimensionality of X
        self.y_dim = y_dim  # dimensionality of Y

        self.sigmaN = sigmaN 
        self.sigmaF = sigmaF
        self.sigmaL = sigmaL

        # data
        self.X = np.zeros([x_dim, indivDataThersh], dtype=float)
        self.Y = np.zeros([y_dim, indivDataThersh], dtype=float)

        self.K = np.zeros([y_dim * indivDataThersh, indivDataThersh], dtype=float)  # covariance matrix
        self.K_border = np.zeros([y_dim, indivDataThersh], dtype=float)
        self.K_corner = np.zeros([y_dim, 1], dtype=float)
        self.KinvKborder = np.zeros([y_dim, indivDataThersh], dtype=float)
        self.priorFunc = priorFunc
        self.priorErrorList = np.array([])
        self.predictTimes = 0

    @ensure_annotations
    def kernel(self, Xi: np.ndarray, Xj: np.ndarray, dimension: int) -> np.ndarray:
        '''
        Only use for one-dimensional prediction
        Args:
            Xi: old_X
            Xj: new_X
        Returns:
            K Matrix
        '''
        if Xi.ndim == 1:  # if only one point
            kernelMatrix = (self.sigmaF[dimension] ** 2) * np.exp(-0.5 * np.sum(((Xi - Xj) / self.sigmaL[:, dimension]) ** 2))
            return kernelMatrix
        else:
            kernlMatrix = np.zeros([Xi.shape[1], Xj.shape[1]], dtype=float)
            for Xj_Nr in range(Xj.shape[1]):
                a = -0.5 * np.sum(
                        ((Xi - np.matlib.repmat(Xj[:,Xj_Nr].reshape(self.x_dim, -1), 1, Xi.shape[1])) / self.sigmaL[:, dimension].reshape(self.x_dim,1)
                        ) ** 2, axis=0)

                kernlMatrix[:, Xj_Nr] = (self.sigmaF[dimension] ** 2) * np.exp(a)  
            return kernlMatrix


    def addDataEntire(self, X_in, Y_in):
        AllDataQuantity = min(X_in.shape[1], self.indivDataThersh)
        self.DataQuantity = AllDataQuantity
        self.X[:, range(self.DataQuantity)] = X_in[:, range(AllDataQuantity)]
        self.Y[:, range(self.DataQuantity)] = Y_in[:, range(AllDataQuantity)]

    def updateKmatEntire(self):
        X_set = self.X[:,range(self.DataQuantity)]
        for i in range(self.y_dim):
            K = self.kernel(X_set, X_set, i)
            K_noise =  K + self.sigmaN[i] ** 2 * np.eye(self.DataQuantity, dtype=float)
            self.K[i*self.indivDataThersh : i*self.indivDataThersh+self.DataQuantity, 
                   0:self.DataQuantity] = K_noise

    def addDataOnce(self, x, y):
        self.X[:, self.DataQuantity] = x
        self.Y[:, self.DataQuantity] = y
        self.DataQuantity = self.DataQuantity + 1

    def errorRecord(self,x,y):
        temp_error = np.absolute(self.priorFunc(x) - y)
        self.priorErrorList = np.append(self.priorErrorList, temp_error)

    def updateKmatOnce(self):
        if self.DataQuantity == 1:
            self.updateKmatEntire
        else:
            x_new = self.X[:, self.DataQuantity-1].reshape(self.x_dim, 1)
            X_old = self.X[:, 0: self.DataQuantity-1]
            for i in range(self.y_dim):
                temp_K_border = self.kernel(X_old, x_new, i)
                self.K_border[[i], 0:self.DataQuantity-1] = temp_K_border.transpose()
                self.K_corner[[i], 0] = self.kernel(x_new, x_new,i) + self.sigmaN[i] ** 2
                K_old = self.K[i*self.indivDataThersh : i*self.indivDataThersh+self.DataQuantity-1,
                       0:self.DataQuantity-1]
                K_new = np.vstack((
                    np.hstack((K_old, self.K_border[[i], 0:self.DataQuantity-1].transpose())),
                    np.hstack((self.K_border[[i], 0:self.DataQuantity-1], self.K_corner[[i], 0].reshape(1,1)))
                ))
                self.K[i*self.indivDataThersh : i*self.indivDataThersh+self.DataQuantity,
                                    0:self.DataQuantity] = K_new

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.DataQuantity == 0:
            mu = np.zeros([self.y_dim, 1], dtype=float)
            var = self.sigmaF ** 2
            return mu, var
        else:
            X_train = self.X[:, 0:self.DataQuantity]
            mu = np.zeros([self.y_dim, 1], dtype=float)
            var = np.zeros([self.y_dim, 1], dtype=float)
            for i_dim in range(self.y_dim):
                temp_K_border = self.kernel(X_train, x, i_dim)
                temp_K = self.K[i_dim*self.indivDataThersh : i_dim*self.indivDataThersh+self.DataQuantity, 
                                0:self.DataQuantity]
                temp_KinvYtrain= np.linalg.solve(temp_K, (self.Y[i_dim, 0:self.DataQuantity] - self.priorFunc(self.X[:, 0: self.DataQuantity]).flatten()) )
                mu[[i_dim], :] = self.priorFunc(x) +  np.dot(temp_KinvYtrain, temp_K_border)
                
                
                temp_KinvKborder = np.linalg.solve(temp_K, temp_K_border)
                var[[i_dim], :] = self.kernel(x,x, i_dim) \
                                        - np.dot(temp_K_border.transpose(), temp_KinvKborder)
                self.K_border[[i_dim], 0:self.DataQuantity] = temp_K_border.transpose()
            return mu, var


    