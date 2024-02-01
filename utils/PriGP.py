# Copyright (c) by Zewen Yang under GPL-3.0 license
# Last modified: Zewen Yang 02/2024

import math
import numpy as np
from scipy.stats import norm
from utils.GPmodel import GPmodel 
from utils.common import * 


class PriGP():
    def __init__(self, x_dim, y_dim, indivDataThersh, 
                 sigmaN, sigmaF, sigmaL, 
                 priorFuncList, agentQuantity, Graph):
        
        self.indivDataThersh = indivDataThersh
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.sigmaN = sigmaN 
        self.sigmaF = sigmaF
        self.sigmaL = sigmaL
        self.priorFunc = priorFuncList
        self.agentQuantity = agentQuantity
        self.agents = []
        self.G = Graph
        self.error_list = [[[] for _ in range(agentQuantity)] for _ in range(agentQuantity)]
        self.neighbors_list = [[] for _ in range(agentQuantity)]
        self.requestNieghborsList = [[] for _ in range(agentQuantity)]
        self.aggWieghtsList = [[] for _ in range(agentQuantity)]
        self.ordered_error_list = [[] for _ in range(agentQuantity)]
        self.largset_indices = [[] for _ in range(agentQuantity)]
        self.MASpredictTimes = [[0] for _ in range(agentQuantity)]
        self.deleteNieghborsQuantity = [1 for _ in range(agentQuantity)]

        for i in range(agentQuantity):
            agent = GPmodel(x_dim, y_dim, indivDataThersh, sigmaN, sigmaF, sigmaL, priorFuncList[i])
            self.agents.append(agent)
            self.neighbors_list[i] = list(self.G.neighbors(i))


    def addAgentsDataEntire(self, X_train, Y_train):
        for i_agent in range(self.agentQuantity):
            self.agents[i_agent].addDataEntire(X_train[i_agent], Y_train[i_agent])

    def updateAgentsKmatEntire(self):
        for i_agent in range(self.agentQuantity):
            self.agents[i_agent].updateKmatEntire()


    def reqestWhichNeighbors(self, i_agent):
        if self.MASpredictTimes[i_agent][0] == 0:
            temp_neighbors_list = self.neighbors_list[i_agent]
            temp_weights = equalProportions(len(self.neighbors_list[i_agent]))
            self.requestNieghborsList[i_agent] = self.neighbors_list[i_agent]
            self.aggWieghtsList[i_agent] = equalProportions(len(self.neighbors_list[i_agent]))
        else:
            temp_neighbors_list = self.neighbors_list[i_agent]
            for i in range(len(temp_neighbors_list)):
                s = temp_neighbors_list[i]
                agent_error = self.agents[s].priorErrorList
                agent_error = np.sum(agent_error)/len(agent_error)
                self.error_list[i_agent][s] = agent_error
            temp_errorlist =  self.error_list[i_agent] 
            non_empty_values = [item for item in temp_errorlist  if not isinstance(item, list) or np.size(item) > 0]
            non_empty_values = minmaxScaling(non_empty_values)
            self.ordered_error_list[i_agent] = non_empty_values
            deleteNieghborsQuantity = self.deleteNieghborsQuantity[i_agent]
            sorted_indices = np.argsort(-np.array(non_empty_values))
            largset_indices = sorted_indices[:deleteNieghborsQuantity].flatten()
            self.largset_indices[i_agent] = largset_indices

            sort_agentList = np.sort(temp_neighbors_list)
            requestNieghborsList = np.delete(sort_agentList,largset_indices, axis=0)
            self.requestNieghborsList[i_agent] = requestNieghborsList
            mean = non_empty_values[largset_indices.item()]
            std_dev = 0.25
            temp_weights = 1/norm.pdf(non_empty_values, loc=mean, scale=std_dev)
            temp_weights = np.delete(temp_weights, largset_indices, axis=0)
            self.aggWieghtsList[i_agent] = temp_weights


    def predict_Pri(self, i_agent, x):
        temp_agents_list = self.requestNieghborsList[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_weights = self.aggWieghtsList[i_agent]
        temp_weights = getProportions(temp_weights)
        temp_mu_list = []
        temp_var_list = []
        weight_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            weight = temp_weights[i]
            mu, var = self.agents[act_agent].predict(x)
            weight_list.append(weight)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(var))
        mu = np.dot(temp_mu_list, weight_list)
        self.MASpredictTimes[i_agent][0] += 1
        return mu
    
    def predict_MoE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_weights = equalProportions(len(temp_agents_list))
        temp_mu_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
        mu = np.dot(temp_mu_list, temp_weights)
        return mu

    def predict_PoE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(1/var))
        mu = np.dot(temp_mu_list, temp_var_list)
        mu = mu/np.sum(temp_var_list)
        return mu

    def predict_gPoE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            beta_k = math.log((self.sigmaF**2 + 1e-5)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/(var)))
        mu = np.dot(temp_mu_list, temp_var_list)
        mu = mu/np.sum(temp_var_list)
        return mu
    
    def predict_gPOE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            beta_k = math.log((self.sigmaF**2 + 1e-5)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/var))
        var_GPOE = np.sum(temp_var_list)
        weights_g = temp_var_list/var_GPOE
        mu_GPOE = np.dot(temp_mu_list, weights_g)
        return mu_GPOE
    
    def predict_BCM(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(1/var))
        prior_var = (1-len(temp_agents_list))/(self.sigmaF**2+ self.sigmaN**2)
        var_BCM = np.sum(temp_var_list)+ prior_var.ravel()
        weights = temp_var_list/var_BCM
        mu_BCM = np.dot(temp_mu_list, weights)
        return mu_BCM
    
    def predict_rBCM(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        beta_k_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            beta_k = math.log((self.sigmaF**2 + self.sigmaN**2)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/var))
            beta_k_list.append(np.squeeze(beta_k))
        bcm_var = (1-np.sum(beta_k_list))/(self.sigmaF**2+ self.sigmaN**2)
        var_RBCM = np.sum(temp_var_list )+ bcm_var.ravel()
        weights = temp_var_list/var_RBCM
        mu_RBCM = np.dot(temp_mu_list, weights)
        return mu_RBCM
    
    def predict_Pri_withVar(self, i_agent, x):
        temp_agents_list = self.requestNieghborsList[i_agent]
        temp_agents_list = np.sort(temp_agents_list)
        temp_weights = self.aggWieghtsList[i_agent]
        temp_mu_list = []
        weight_list = []
        for i in range(len(temp_agents_list)):
            act_agent = temp_agents_list[i]
            mu, var = self.agents[act_agent].predict(x)
            temp_mu_list.append(np.squeeze(mu))
            weight = ((temp_weights[i])**(1/2)) * ((1/var)**(1/2))
            weight_list.append(np.squeeze(weight))
        weights = getProportions(weight_list)
        mu = np.dot(temp_mu_list, weights)
        self.MASpredictTimes[i_agent][0] += 1
        return mu

