# Copyright (c) by Zewen Yang under GPL-3.0 license
# Last modified: Zewen Yang 02/2024

from utils.PriGP import PriGP
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import math
import networkx as nx
from scipy.stats import norm

# Create an empty graph
G = nx.Graph()
# Add 4 nodes to the graph
nodes = range(0, 4)
G.add_nodes_from(nodes)
# Add self-loops to each node
for node in nodes:
    G.add_edge(node, node)
# Manually specify the edges you want in your graph
edges = [(0, 2), (1, 0), (1, 3), (3, 2)]
# Add the specified edges to the graph
G.add_edges_from(edges)
x_dim = 1
y_dim = 1
indivDataThersh = 100
sigmaN = 0.1 * np.ones([y_dim, 1], dtype=float)
sigmaF = 1.0 * np.ones([y_dim, 1], dtype=float)
sigmaL = 0.2 * np.ones([x_dim, y_dim], dtype=float)
agentQuantity = 4
priorFunc2 = lambda x: -1*np.ones([1,np.size(x,1)])
priorFunc3 = lambda x: 1*np.sin(2*x)
priorFunc4 = lambda x: 1*np.cos(2*x)
priorFunc1 = lambda x: 0*np.ones([1,np.size(x,1)])
priorFunc_lsit =[priorFunc1,priorFunc2,priorFunc3,priorFunc4]

np.random.seed(42)
dataQuantity_train = 10
lower_range, upper_range = 0, 2*math.pi
X_train_list = []
Y_true_list = []
Y_train_list = []

X_train = (upper_range - lower_range) * np.random.rand(x_dim, dataQuantity_train) + lower_range
Y_true = np.array([np.sin(2*X_train[0,:])])
Y_train = Y_true + 10e-4*np.random.normal(loc=0.0, scale=0.05, size=dataQuantity_train)
for _ in range(agentQuantity):
    X_train_list.append(X_train)
    Y_true_list.append(Y_true)
    Y_train_list.append(Y_train)
    
dataQuantity_test = 1000
X_test = np.array([np.linspace(lower_range, upper_range, dataQuantity_test)])
Y_test = np.array([np.sin(2*X_test[0,:])])
X_test_prior1 = priorFunc1(X_test)
X_test_prior2 = priorFunc2(X_test)
X_test_prior3 = priorFunc3(X_test)
X_test_prior4 = priorFunc4(X_test)

# PreTraining
PRIGP = PriGP(x_dim, y_dim, indivDataThersh, 
                    sigmaN, sigmaF, sigmaL, 
                    priorFunc_lsit, agentQuantity, G)
PRIGP.addAgentsDataEntire(X_train_list, Y_train_list)
PRIGP.updateAgentsKmatEntire()
for j_data in range(dataQuantity_train):
    for i_agent in range(agentQuantity):
        x = X_train_list[i_agent][:, [j_data]]
        y = Y_train_list[i_agent][:,[j_data]]
    for i in range(agentQuantity):
        PRIGP.agents[i].errorRecord(x,y)

# Testing
muPriVar_test_list = [[] for _ in range(agentQuantity)]
muPri_test_list = [[] for _ in range(agentQuantity)]
muInd_test_list = [[] for _ in range(agentQuantity)]
varInd_test_list = [[] for _ in range(agentQuantity)]
varInd_sd_test_list = [[] for _ in range(agentQuantity)]
muPOE_test_list = [[] for _ in range(agentQuantity)]
muGPOE_test_list = [[] for _ in range(agentQuantity)]
muBCM_test_list = [[] for _ in range(agentQuantity)]
muRBCM_test_list = [[] for _ in range(agentQuantity)]
muMOE_test_list = [[] for _ in range(agentQuantity)]

for j_data in range(X_test.shape[1]):
    for i_agent in range(agentQuantity):
        x = X_test[:, [j_data]]
        PRIGP.reqestWhichNeighbors(i_agent)
        mu_Pri = PRIGP.predict_Pri(i_agent, x)
        mu_PriVar = PRIGP.predict_Pri_withVar(i_agent, x)
        mu_POE = PRIGP.predict_PoE(i_agent, x)
        mu_gPOE = PRIGP.predict_gPoE(i_agent, x)
        mu_BCM = PRIGP.predict_BCM(i_agent, x)
        mu_rBCM = PRIGP.predict_rBCM(i_agent, x)
        mu_MOE = PRIGP.predict_MoE(i_agent, x)
        mu_ind, var_ind = PRIGP.agents[i_agent].predict(x)
        muMOE_test_list[i_agent].append(mu_MOE)
        muPOE_test_list[i_agent].append(mu_POE)
        muGPOE_test_list[i_agent].append(mu_gPOE)
        muBCM_test_list[i_agent].append(mu_BCM)
        muRBCM_test_list[i_agent].append(mu_rBCM)
        muPri_test_list[i_agent].append(mu_Pri)
        muPriVar_test_list[i_agent].append(mu_PriVar)
        muInd_test_list[i_agent].append(np.squeeze(mu_ind))
        varInd_test_list[i_agent].append(var_ind)

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

linewidth=1.5
alpha=0.6
fig1, axs1 = plt.subplots(2,2)

axs1[0,0].plot(X_train_list[0][0,:], Y_train_list[0][0,:], 'o', markersize=5, markerfacecolor='none', markeredgecolor='#fb0c0d')
axs1[0,0].plot(X_test[0,:], Y_test[0,:], '-', color = "#fb0c0d", linewidth=linewidth)
axs1[0,0].plot(X_test[0,:], X_test_prior1[0,:], 'k-', linewidth=linewidth, alpha=alpha)
axs1[0,0].plot(X_test[0,:], np.squeeze(muPOE_test_list[0][:]), '-', color = '#89ba16', linewidth=linewidth)
axs1[0,0].plot(X_test[0,:], np.squeeze(muPri_test_list[0][:]), linestyle='--', color = '#2077b4', linewidth=linewidth)

axs1[0,1].plot(X_train_list[1][0,:], Y_train_list[1][0,:],  'o', markersize=5, markerfacecolor='none', markeredgecolor='#fb0c0d')
axs1[0,1].plot(X_test[0,:], Y_test[0,:], '-', color = "#fb0c0d",linewidth=linewidth)
axs1[0,1].plot(X_test[0,:], X_test_prior2[0,:], 'k-', linewidth=linewidth, alpha=alpha)
axs1[0,1].plot(X_test[0,:], np.squeeze(muPOE_test_list[1][:]), '-', color = '#89ba16', linewidth=linewidth)
axs1[0,1].plot(X_test[0,:], np.squeeze(muPri_test_list[1][:]), linestyle='--', color = '#2077b4', linewidth=linewidth)

axs1[1,0].plot(X_train_list[2][0,:], Y_train_list[2][0,:],  'o', markersize=5, markerfacecolor='none', markeredgecolor='#fb0c0d')
axs1[1,0].plot(X_test[0,:], Y_test[0,:], '-', color = "#fb0c0d", linewidth=linewidth)
axs1[1,0].plot(X_test[0,:], X_test_prior3[0,:], 'k-', linewidth=linewidth, alpha=alpha)
axs1[1,0].plot(X_test[0,:], np.squeeze(muPOE_test_list[2][:]), '-', color = '#89ba16', linewidth=linewidth)
axs1[1,0].plot(X_test[0,:], np.squeeze(muPri_test_list[2][:]), linestyle='--', color = '#2077b4', linewidth=linewidth)

axs1[1,1].plot(X_train_list[3][0,:], Y_train_list[3][0,:],  'o', markersize=5, markerfacecolor='none', markeredgecolor='#fb0c0d')
axs1[1,1].plot(X_test[0,:], Y_test[0,:], '-', color = "#fb0c0d", linewidth=linewidth)
axs1[1,1].plot(X_test[0,:], X_test_prior4[0,:], 'k-', linewidth=linewidth, alpha=alpha)
axs1[1,1].plot(X_test[0,:], np.squeeze(muPOE_test_list[3][:]), '-', color = '#89ba16', linewidth=linewidth)
axs1[1,1].plot(X_test[0,:], np.squeeze(muPri_test_list[3][:]), linestyle='--', color = '#2077b4', linewidth=linewidth)

axs1[0,0].yaxis.grid(True)
axs1[0,1].yaxis.grid(True)
axs1[1,0].yaxis.grid(True)
axs1[1,1].yaxis.grid(True)
axs1[0,0].tick_params(axis='x', labelsize=14)  
axs1[0,0].tick_params(axis='y', labelsize=14)
axs1[0,1].tick_params(axis='x', labelsize=14)  
axs1[0,1].tick_params(axis='y', labelsize=14)  
axs1[1,0].tick_params(axis='x', labelsize=14)  
axs1[1,0].tick_params(axis='y', labelsize=14)  
axs1[1,1].tick_params(axis='x', labelsize=14)  
axs1[1,1].tick_params(axis='y', labelsize=14) 
axs1[0,0].set_xticks(np.arange(min(X_test[0,:]), max(X_test[0,:])+0.5, 1))  # Adjust the range and step as needed
axs1[0,0].set_yticks(np.arange(min(Y_test[0,:]), max(Y_test[0,:])+0.5, 0.5))  # Adjust the range and step as needed
axs1[0,1].set_xticks(np.arange(min(X_test[0,:]), max(X_test[0,:])+0.5, 1))
axs1[0,1].set_yticks(np.arange(min(Y_test[0,:]), max(Y_test[0,:])+0.5, 0.5))
axs1[1,0].set_xticks(np.arange(min(X_test[0,:]), max(X_test[0,:])+0.5, 1))
axs1[1,0].set_yticks(np.arange(min(Y_test[0,:]), max(Y_test[0,:])+0.5, 0.5))
axs1[1,1].set_xticks(np.arange(min(X_test[0,:]), max(X_test[0,:])+0.5, 1))
axs1[1,1].set_yticks(np.arange(min(Y_test[0,:]), max(Y_test[0,:])+0.5, 0.5)) 

axs1[0,0].set_ylabel(r"$\tilde{f}(x)$", fontsize=14)
axs1[1,0].set_xlabel("$x$", fontsize=14) 
axs1[1,0].set_ylabel(r"$\tilde{f}(x)$", fontsize=14)
axs1[1,1].set_xlabel("$x$", fontsize=14) 
legend_labels = [r"Training points", r'True', r'Prior', r'POE', r'Pri-GP(c=1)']
fig1.legend(labels = legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=12, ncol=66)
plt.rcParams['figure.figsize'] = [16, 4]
plt.tight_layout()
plt.show()

error_PriVar = np.abs(Y_test - np.array(muPriVar_test_list))
sum_error_PriVar = np.sum(error_PriVar, axis=0)
error_Pri = np.abs(Y_test - np.array(muPri_test_list))
sum_error_Pri = np.sum(error_Pri, axis=0)
error_POE = np.abs(Y_test - np.array(muPOE_test_list))
sum_error_POE = np.sum(error_POE, axis=0)
error_GPOE = np.abs(Y_test - np.array(muGPOE_test_list))
sum_error_GPOE = np.sum(error_GPOE, axis=0)
error_BCM = np.abs(Y_test - np.array(muBCM_test_list))
sum_error_BCM = np.sum(error_BCM, axis=0)
error_RBCM = np.abs(Y_test - np.array(muRBCM_test_list))
sum_error_RBCM = np.sum(error_RBCM, axis=0)
error_MOE = np.abs(Y_test - np.array(muMOE_test_list))
sum_error_MOE = np.sum(error_MOE, axis=0)
error_Ind = np.abs(Y_test - np.array(muInd_test_list))
sum_error_Ind = np.sum(error_Ind, axis=0)

print("Pri:",np.sum(error_Pri, axis=1)/1000)
print("PriVar:" ,np.sum(error_PriVar, axis=1)/1000)
print("POE:" ,np.sum(error_POE, axis=1)/1000)
print("GPOE:" ,np.sum(error_GPOE, axis=1)/1000)
print("BCM:" ,np.sum(error_BCM, axis=1)/1000)
print("RBCM:" ,np.sum(error_RBCM, axis=1)/1000)
print("MOE:" ,np.sum(error_MOE, axis=1)/1000)
print("Ind:" ,np.sum(error_Ind, axis=1)/1000)

from matplotlib.ticker import FixedLocator, FixedFormatter
data = [sum_error_Pri,sum_error_PriVar, sum_error_POE, sum_error_GPOE, sum_error_BCM,sum_error_RBCM,sum_error_MOE, sum_error_Ind,]
labels = [f"Pri-GP\n(c=1)", f'Pri-GP\n(c=0.5)', r'POE',r'GPOE', r'BCM',r'RBCM', r'MOE', r'IGP']
custom_ticks = [1, 2, 3, 4, 5,6,7,8]

fig, axs = plt.subplots()
violin = axs.violinplot(data, showmeans = True, showmedians = False)
vp = violin['cmeans']
vp.set_edgecolor("red")
vp.set_linewidth(1.5)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
axs.set_ylabel(r'${\Delta e}$', fontsize=12)
plt.yticks(fontsize=12)
axs.xaxis.set_major_locator(FixedLocator(custom_ticks))
axs.xaxis.set_major_formatter(FixedFormatter(labels))
axs.yaxis.grid(True)
axs.xaxis.grid(True)
axs.tick_params(axis='x', labelsize=12)  
axs.tick_params(axis='y', labelsize=12)  
plt.rcParams['figure.figsize'] = [8, 4]
plt.show()
