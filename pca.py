# Jessica Wei
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_reader import *

# Swing Phase PCA
def getPrincipleComponentsSP(n, data, dataIndex):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 

    # Get the sample name
    sample = np.transpose(np.array(data[dataIndex][0]))
    
    # Seperating out the targets
    wl = np.transpose(np.array(data[dataIndex][1]))
    wr = np.transpose(np.array(data[dataIndex][2]))
    leftSwingStart = wl[0][0] < wr[0][0]
    # Get phase transition times
    phaseTimes = []
    if leftSwingStart == True:
        for i,j in zip(wl,wr):
            phaseTimes.append(i[0])
            phaseTimes.append(i[1])
            phaseTimes.append(j[0])
            phaseTimes.append(j[1])
        if wl.shape[0] > wr.shape[0]:
            phaseTimes.append(wl[wl.shape[0]-1][0])
            phaseTimes.append(wl[wl.shape[0]-1][1])
    else:
        for i,j in zip(wr,wl):
            phaseTimes.append(i[0])
            phaseTimes.append(i[1])
            phaseTimes.append(j[0])
            phaseTimes.append(j[1])
        if wr.shape[0] > wl.shape[0]:
            phaseTimes.append(wr[wr.shape[0]-1][0])
            phaseTimes.append(wr[wr.shape[0]-1][1])

    targets = ['Left Swing', 'Left Heel', 'Right Swing', 'Right Heel']
    colors = ['r','g','b','c']
    y = []
    timestamps = get_timestamps(dataIndex)
    # Swing = even, Heel = odd
    for times in range(timestamps.size):
        targetIndex = 0
        for checkInd in range(len(phaseTimes)):
            if timestamps[0][times] <= phaseTimes[checkInd] + 1:
                targetIndex -= 1
                break
            targetIndex+=1
        if leftSwingStart == True:
            y.append(targets[targetIndex%4])
        else:
            y.append(targets[(targetIndex+2)%4])
    yDF = pd.DataFrame(y, columns = ['target'])

    # Separating out the features
    x = np.transpose(np.array(data[dataIndex][3]))
    
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    # Projecting to n-D through PCA
    #p = PCA(n_components = int(n))
    p = PCA(n_components = n).fit(x)
    print(p.explained_variance_ratio_)
    
    p = PCA(n_components = int(n)).fit(x)
    principalComponents = p.fit_transform(x)
    
    principaldf = pd.DataFrame(data = principalComponents, 
                               columns = ['principal component 1', 'principal component 2'])
    
    # Add the feature list back onto the PCA'd list
    finalDf = pd.concat([principaldf, yDF], axis = 1)
    
    # Plot the points
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)

    # Draw the graph
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    trailName = str(sample)
    sample = str(sample) + ": " + str(round((p.explained_variance_ratio_[0] + p.explained_variance_ratio_[1])*100,2)) +"%"
    ax.set_title(sample, fontsize = 20)
    ax.legend(targets)
    ax.grid()
    
    saveLocation = 'FiguresPCA'
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    plt.savefig(os.path.join(saveLocation, trailName + " PCA" + ".png"))
    
data = read_data()
getPrincipleComponentsSP(2, data, 0)

# If you want all the graphs loaded, please uncomment the code below
# for i in range(len(data)):
#     getPrincipleComponentsSP(2, data, i)



