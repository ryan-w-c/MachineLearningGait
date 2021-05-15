# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:57:36 2021

@author: Graysire
"""

from data_reader import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.interpolate as spi


#get data
data = read_data()

#create tsne
tsne = TSNE(n_components=2)

#define and create save location
saveLocation = 'FiguresTSNE3'
    
if not os.path.exists(saveLocation):
    os.makedirs(saveLocation)

'''
#used for restructuring data when regualrizing data size for collated use

#determine the standard size based on the smallest dataset
min = 9999999999

for d in data:
    
    if len(d[3][0]) < min:
        min = len(d[3][0])

# restructure data
dataT = np.zeros((len(data),12,min))

#for each object and sensor
for dataObj in range(len(data)):
    #print(len(dataObj[0]))
    for sensor in range(12):
        #create linear space equating to entry number
        linspace1 = np.linspace(1, len(data[dataObj][3][sensor]), num=len(data[dataObj][3][sensor]))
        #create interpolation on the data to allow for resizing
        interp = spi.interp1d(linspace1, data[dataObj][3][sensor])

        #create resized linear space and put it into restructured data
        linspace = np.linspace(1,min,num=min,endpoint=True)
        dataT[dataObj,sensor] = interp(linspace)
        #plt.plot(linspace, interp(linspace))
        
#plot experiment+sensor against each other

#reshape data so each row represents experiment+sensor and each column is the data values
dataTR = np.reshape(dataT,(len(dataT)*12,min))

#perfom t-sne
scatterPlot = tsne.fit_transform(dataTR)

#plot the data
for i in range(0,min,12):
    for a in range(12):
        try:
            plt.scatter(scatterPlot[i+a,0],scatterPlot[i+a,1],c=clist[a])
            # print("a")
        except:
            break

#create the legend and save the file
plt.legend(llist, loc='upper left', title='Legend', frameon=True, fontsize='large', bbox_to_anchor=(1,1.1))
plt.savefig("tsne.png")
'''




#print(data[0][0])
#print(np.transpose(data[0][1]))
#print(np.transpose(data[0][2]))


#print(np.sort(np.concatenate((np.transpose(data[0][1]), np.transpose(data[0][2]))), axis=0))

'''
#perform t-sne on all the data
for i in range(len(data)):
    #takes WR and WL start/stop times and combines into array of size [n,2] sorted ascending on the columns
    #ex. [[763,867],[898,943]] from [[898,943]] and [[763,867]]
    startStops = np.sort(np.concatenate((np.transpose(data[i][1]), np.transpose(data[i][2]))), axis=0)
    #adjust values for truncated data
    startStops -= startStops[0,0]

    #creates t-sne data points using each entry as object and sensor values as features
    points = tsne.fit_transform(np.transpose(np.array(data[i][3])))

    #create initial swing and stance point arrays
    swingPoints = points[startStops[0,0]:startStops[0,1] + 1]
    stancePoints = points[startStops[0,1] + 1:startStops[1,0]]
    
    # print(len(points))
    # print("Initial Swing Count: " + str(len(swingPoints)))
    for a in range(len(startStops) - 1):
        # print("Additional Points: " + str(len(points[startStops[a + 1,0] : startStops[a + 1,1] + 1])))
        swingPoints = np.concatenate((swingPoints, points[startStops[a + 1,0] : startStops[a + 1,1] + 1]))
        # print("New Swing Count: " + str(len(swingPoints)))
        if a + 2 < len(startStops):
            stancePoints = np.concatenate((stancePoints, points[startStops[a + 1,1] + 1 : startStops[a + 2,0]]))
        else:
            stancePoints = np.concatenate((stancePoints, points[startStops[a + 1,1] + 1 :]))
    
    # print(len(swingPoints))
    # print(len(stancePoints))
    plt.clf()
    plt.scatter(swingPoints[:,0],swingPoints[:,1],label="Swing",s=13)
    plt.scatter(stancePoints[:,0],stancePoints[:,1],label="Stance",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    plt.savefig("FiguresTSNE\\" + data[i][0] + " tsne.png")
'''

'''
#perform t-sne on all experiment-sensors labelling only swing and stance
for i in range(len(data)):
    #takes WR and WL start/stop times and combines into array of size [n,2] sorted ascending on the columns
    #ex. [[763,867],[898,943]] from [[898,943]] and [[763,867]]
    startStops = np.sort(np.concatenate((np.transpose(data[i][1]), np.transpose(data[i][2]))), axis=0)
    #adjust values for truncated data
    startStops -= startStops[0,0]

    #creates t-sne data points using each entry as object and sensor values as features
    points = tsne.fit_transform(np.transpose(np.array(data[i][3])))

    #create initial swing and stance point arrays
    swingPoints = points[startStops[0,0]:startStops[0,1] + 1]
    stancePoints = points[startStops[0,1] + 1:startStops[1,0]]

    for a in range(len(startStops) - 1):
        swingPoints = np.concatenate((swingPoints, points[startStops[a + 1,0] : startStops[a + 1,1] + 1]))
        if a + 2 < len(startStops):
            stancePoints = np.concatenate((stancePoints, points[startStops[a + 1,1] + 1 : startStops[a + 2,0]]))
        else:
            stancePoints = np.concatenate((stancePoints, points[startStops[a + 1,1] + 1 :]))
    
    # print(len(swingPoints))
    # print(len(stancePoints))
    plt.clf()
    plt.scatter(swingPoints[:,0],swingPoints[:,1],label="Swing",s=13)
    plt.scatter(stancePoints[:,0],stancePoints[:,1],label="Stance",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    plt.savefig("FiguresTSNE\\" + data[i][0] + " tsne.png")
'''

'''

#IMPORTANT NOTE: The assumptions about the data include the following
#   1. Steps alternate between right and left
#   2. Due to the above, len(rightSteps) - len(leftSteps) is -1, 0, or 1


#performs t-sne on all experiment sensors with labelled distinction between left and right foot stance and swings
for i in range(len(data)):
    #get sorted array of starts and stops from WL and WR
    startStops_left = np.sort(np.transpose(data[i][1]), axis = 0)
    startStops_right = np.sort(np.transpose(data[i][2]), axis = 0)
    #get the minimum value to adjust indexing
    startMin = min(startStops_left[0,0], startStops_right[0,0])
    #adjsut the arrays for indexing the points generated by t-sne
    startStops_left -= startMin
    startStops_right -= startMin
    
    #get the t-sne points when all points are compared
    points_total = tsne.fit_transform(np.transpose(np.array(data[i][3])))
    
    #Initialize swingPoints as start to stop inclusive
    swingPoints_left = points_total[startStops_left[0,0]:startStops_left[0,1] + 1]
    swingPoints_right = points_total[startStops_right[0,0]:startStops_right[0,1] + 1]
    
    #if the left foot moves first, left stance stop X goes to right stance start X
    #otherwise it left stance stop X goes to right stance start X + 1 and vice versa for right stance
    if startStops_left[0,0] < startStops_right[0,0]:
        stancePoints_left = points_total[startStops_left[0,1] + 1:startStops_right[0,0]]
        stancePoints_right = points_total[startStops_right[0,1] + 1:startStops_left[1,0]]
    else:
        stancePoints_left = points_total[startStops_left[0,1] + 1:startStops_right[1,0]]
        stancePoints_right = points_total[startStops_right[0,1] + 1:startStops_left[0,0]]
    
    for a in range(min(len(startStops_left), len(startStops_right)) - 1):
        #concatenate swing points
        swingPoints_left = np.concatenate((swingPoints_left, points_total[startStops_left[a + 1,0] : startStops_left[a + 1,1] + 1]))
        swingPoints_right = np.concatenate((swingPoints_right, points_total[startStops_right[a + 1,0] : startStops_right[a + 1,1] + 1]))
        
        #if left foot starts
        if startStops_left[0,0] < startStops_right[0,0]:
            #if it is valid to do so, concatenate right stance points
            if a + 2 < len(startStops_left):
                stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 1,1] + 1 : startStops_left[a + 2,0]]))
                #if this is true, concatenate last left stance and swing points otherwise they will be missed
                if a + 3 == len(startStops_left) and len(startStops_left) > len(startStops_right):
                    stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 2,1] + 1 : ]))
                    swingPoints_left = np.concatenate((swingPoints_left, points_total[startStops_left[a + 2,0] : startStops_left[a + 2,1] + 1]))
            #if there are no other left starts then whatever points remain at the end are right stance
            else:
                stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 1,1] + 1 : ]))
            #concantenate left stance
            stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 1,1] + 1 : startStops_right[a + 1,0]]))
        #if right foot starts
        else:
            #if it is valid to do so, concatenate left stance points
            if a + 2 < len(startStops_right):
                stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 1,1] + 1 : startStops_right[a + 2,0]]))
                #if this is true, concatenate last right stance and swing points otherwise they will be missed
                if a + 3 == len(startStops_right) and len(startStops_right) > len(startStops_left):
                    stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 2,1] + 1 : ]))
                    swingPoints_right = np.concatenate((swingPoints_right, points_total[startStops_right[a + 2,0] : startStops_right[a + 2,1] + 1]))
            #if there are no other right starts then whatever points remain at the end are left stance
            else:
                stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 1,1] + 1 : ]))
            #concantenate right stance
            stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 1,1] + 1 : startStops_left[a + 1,0]]))
    plt.clf()
    plt.scatter(swingPoints_left[:,0],swingPoints_left[:,1],label="Left Swing",s=13)
    plt.scatter(stancePoints_left[:,0],stancePoints_left[:,1],label="Left Stance",s=13)
    plt.scatter(swingPoints_right[:,0],swingPoints_right[:,1],label="Right Swing",s=13)
    plt.scatter(stancePoints_right[:,0],stancePoints_right[:,1],label="Right Stance",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    
    plt.savefig(saveLocation + "\\" + data[i][0] + " tsne.png")


#performs t-sne on all experiment sensors seperating right and left feet
for i in range(len(data)):
    #get sorted array of starts and stops from WL and WR
    startStops_left = np.sort(np.transpose(data[i][1]), axis = 0)
    startStops_right = np.sort(np.transpose(data[i][2]), axis = 0)
    #get the minimum value to adjust indexing
    startMin = min(startStops_left[0,0], startStops_right[0,0])
    #adjsut the arrays for indexing the points generated by t-sne
    startStops_left -= startMin
    startStops_right -= startMin
    
    #get the t-sne points when all points are compared
    points_total = np.transpose(np.array(data[i][3]))
    
    #Initialize swingPoints as start to stop inclusive
    swingPoints_left = points_total[startStops_left[0,0]:startStops_left[0,1] + 1,0:6]
    swingPoints_right = points_total[startStops_right[0,0]:startStops_right[0,1] + 1,6:12]
    
    #if the left foot moves first, left stance stop X goes to right stance start X
    #otherwise it left stance stop X goes to right stance start X + 1 and vice versa for right stance
    if startStops_left[0,0] < startStops_right[0,0]:
        stancePoints_left = points_total[startStops_left[0,1] + 1:startStops_right[0,0],0:6]
        stancePoints_right = points_total[startStops_right[0,1] + 1:startStops_left[1,0],6:12]
    else:
        stancePoints_left = points_total[startStops_left[0,1] + 1:startStops_right[1,0],0:6]
        stancePoints_right = points_total[startStops_right[0,1] + 1:startStops_left[0,0],6:12]
    
    for a in range(min(len(startStops_left), len(startStops_right)) - 1):
        #concatenate swing points
        swingPoints_left = np.concatenate((swingPoints_left, points_total[startStops_left[a + 1,0] : startStops_left[a + 1,1] + 1,0:6]))
        swingPoints_right = np.concatenate((swingPoints_right, points_total[startStops_right[a + 1,0] : startStops_right[a + 1,1] + 1,6:12]))
        
        #if left foot starts
        if startStops_left[0,0] < startStops_right[0,0]:
            #if it is valid to do so, concatenate right stance points
            if a + 2 < len(startStops_left):
                stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 1,1] + 1 : startStops_left[a + 2,0],6:12]))
                #if this is true, concatenate last left stance and swing points otherwise they will be missed
                if a + 3 == len(startStops_left) and len(startStops_left) > len(startStops_right):
                    stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 2,1] + 1 : ,0:6]))
                    swingPoints_left = np.concatenate((swingPoints_left, points_total[startStops_left[a + 2,0] : startStops_left[a + 2,1] + 1,0:6]))
            #if there are no other left starts then whatever points remain at the end are right stance
            else:
                stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 1,1] + 1 : ,6:12]))
            #concantenate left stance
            stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 1,1] + 1 : startStops_right[a + 1,0],0:6]))
        #if right foot starts
        else:
            #if it is valid to do so, concatenate left stance points
            if a + 2 < len(startStops_right):
                stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 1,1] + 1 : startStops_right[a + 2,0],0:6]))
                #if this is true, concatenate last right stance and swing points otherwise they will be missed
                if a + 3 == len(startStops_right) and len(startStops_right) > len(startStops_left):
                    stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 2,1] + 1 : ,6:12]))
                    swingPoints_right = np.concatenate((swingPoints_right, points_total[startStops_right[a + 2,0] : startStops_right[a + 2,1] + 1,6:12]))
            #if there are no other right starts then whatever points remain at the end are left stance
            else:
                stancePoints_left = np.concatenate((stancePoints_left, points_total[startStops_left[a + 1,1] + 1 : ,0:6]))
            #concantenate right stance
            stancePoints_right = np.concatenate((stancePoints_right, points_total[startStops_right[a + 1,1] + 1 : startStops_left[a + 1,0],6:12]))

    #create left and right foot t-sne
    points_left = tsne.fit_transform(np.concatenate((swingPoints_left,stancePoints_left)))
    points_right = tsne.fit_transform(np.concatenate((swingPoints_right,stancePoints_right)))

    #print(len(points_total[i]))
    plt.clf()
    plt.scatter(points_left[0:len(swingPoints_left),0],points_left[0:len(swingPoints_left),1],label="Left Swing",s=13)
    plt.scatter(points_left[len(swingPoints_left):,0],points_left[len(swingPoints_left):,1],label="Left Stance",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    plt.savefig(saveLocation + "\\" + data[i][0] + " left foot tsne.png")
    
    plt.clf()
    plt.scatter(points_right[0:len(swingPoints_right),0],points_right[0:len(swingPoints_right),1],label="Right Swing",s=13)
    plt.scatter(points_right[len(swingPoints_right):,0],points_right[len(swingPoints_right):,1],label="Right Stance",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    plt.savefig(saveLocation + "\\" + data[i][0] + " right foot tsne.png")

'''

# restructure data
# dataT = np.zeros((len(data),12,270))

#for each object and sensor
for dataObj in range(1):
    #print(len(dataObj[0]))
    for sensor in range(12):
        #create linear space equating to entry number
        linspace1 = np.linspace(1, len(data[dataObj][3][sensor]), num=len(data[dataObj][3][sensor]))
        #create interpolation on the data to allow for resizing
        interp = spi.interp1d(linspace1, data[dataObj][3][sensor])

        #create resized linear space and put it into restructured data
        linspace = np.linspace(1,270,num=270,endpoint=True)
        #dataT[dataObj,sensor] = interp(linspace)
        plt.plot(range(1,len(data[dataObj][3][sensor]) + 1), data[dataObj][3][sensor])
        plt.plot(linspace, interp(linspace))
        plt.show()







    # print(len(swingPoints_left))
    # print(len(stancePoints_left))
    # print(len(swingPoints_right))
    # print(len(stancePoints_right))
    # print(len(points_total))

#list of colors to be used and their corresponding sensor labels
clist = ["firebrick","red","salmon","darkgoldenrod","goldenrod","gold","darkgreen","limegreen","mediumseagreen","mediumblue","steelblue","deepskyblue"]
llist = ["FSR1","FSR2","FSR4","FSR5","FSR6","FSR7","FSR8","FSR9","FSR10","FSR11","FSR12","FSR13"]


