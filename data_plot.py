# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:50:20 2021

@author: Graysire
"""
from data_reader2 import *
import matplotlib.pyplot as plt


'''
parameters
    Name - Title of the graph and file name
    entryNum - the entry number, used as x axis passed as [1:N] array
    gaitData - the data to plot on the graph used as y axis passed as [D:N] array
    highlightTime - the toe off or heel strike passed as [A] array of entry numbers
    colorSet - colors to be used for lines and points on the figure passed as [D] array of colors
    markerSize - the size of points on the graph such as toe off or heel strike
    labels - labels used for the legend
    alternate - alternate putting annotations on top and on bottom
    figS - size the output figure
    
    plotsthe data and saves it to a figure
'''
def plot_gait_data(name, entryNum, gaitData, highlightTime, colorSet, markerSize, labels, alternate=False, figS=(15,5)):
    plt.clf()
    plt.figure(figsize=figS)
    
    alt = False
    
    # print(gaitData)
    
    for i in range(len(gaitData[0])):
        plt.plot(entryNum, gaitData[:,i], c=colorSet[i], label=labels[i])
        #print(name)
        plt.scatter(highlightTime, [gaitData[int(x),i] for x in highlightTime], marker='*', c=colorSet[i], s=markerSize)
    
    for x in highlightTime:
        lbl = "entry-" + str(entryNum[int(x)])
        plt.annotate(lbl, (x - len(lbl), plt.axis()[2 + int(alt)] - int(alt) * 0.2))
        if(alternate):
            alt = not alt
    
    plt.title(name)
    plt.legend(loc='upper left', title='Legend', frameon=True, fontsize='large')
    
    # [print(i) for i in highlightTime]
    #plt.show()
    plt.savefig("FiguresTest\\" + name + ".png")
    plt.close()
    

#size used for markers
MARKER_SIZE = 50
#color sets used for graphs
LEFT_HEEL_COLORS = ["firebrick","red","salmon"]
LEFT_TOE_COLORS = ["darkgoldenrod","goldenrod","gold"]
RIGHT_HEEL_COLORS = ["darkgreen","limegreen","mediumseagreen"]
RIGHT_TOE_COLORS = ["mediumblue","steelblue","deepskyblue"]

LEFT_HEEL_LABELS = ["FSR1-heel", "FSR2-heel", "FSR4-heel"]
LEFT_TOE_LABELS = ["FSR5-toe", "FSR6-toe", "FSR7-toe"]
RIGHT_HEEL_LABELS = ["FSR8-heel","FSR9-heel","FSR10-heel"]
RIGHT_TOE_LABELS = ["FSR11-toe","FSR12-toe","FSR13-toe"]


data = get_all_data_truncated(os.path.join(os.getcwd(), "WalkEvenData"))

#data[tuplenumber][name/PS/WL/WR][arrayData]
#plot_gait_data(data[0][0] + "-LeftHeel", data[0][1][:,0], data[0][1][:,2:5], [(x - 1) for x in data[0][2][:,3]], LEFT_HEEL_COLORS, MARKER_SIZE)

# print(data[0][2][:,3])
for tuple in data:
    
    #individual figures
    plot_gait_data(tuple[0] + "-LeftHeel", tuple[1][:,0], tuple[1][:,2:5], [(x - 1) for x in tuple[2][:,3]], LEFT_HEEL_COLORS, MARKER_SIZE, LEFT_HEEL_LABELS)
    plot_gait_data(tuple[0] + "-LeftToe", tuple[1][:,0], tuple[1][:,5:8], [(x - 1) for x in tuple[2][:,2]], LEFT_TOE_COLORS, MARKER_SIZE, LEFT_TOE_LABELS)
    plot_gait_data(tuple[0] + "-RightHeel", tuple[1][:,0], tuple[1][:,8:11], [(x - 1) for x in tuple[3][:,3]], RIGHT_HEEL_COLORS, MARKER_SIZE, RIGHT_HEEL_LABELS)
    plot_gait_data(tuple[0] + "-RightToe", tuple[1][:,0], tuple[1][:,11:14], [(x - 1) for x in tuple[3][:,2]], RIGHT_TOE_COLORS, MARKER_SIZE, RIGHT_TOE_LABELS)
    
    #foot summary figures
    plot_gait_data(tuple[0] + "-LeftFoot", tuple[1][:,0], tuple[1][:,2:8], [(x - 1) for x in tuple[2][:,2:4].flatten()], LEFT_HEEL_COLORS + LEFT_TOE_COLORS, MARKER_SIZE, LEFT_HEEL_LABELS + LEFT_TOE_LABELS, True)
    plot_gait_data(tuple[0] + "-RightFoot", tuple[1][:,0], tuple[1][:,8:14], [(x - 1) for x in tuple[3][:,2:4].flatten()], RIGHT_HEEL_COLORS + RIGHT_TOE_COLORS, MARKER_SIZE, RIGHT_HEEL_LABELS + RIGHT_TOE_LABELS, True)
    
    #summary figure
    plot_gait_data(tuple[0] + "-Summary", tuple[1][:,0], tuple[1][:,2:14], [(x - 1) for x in tuple[2][:,2:4].flatten()] + [(x - 1) for x in tuple[3][:,2:4].flatten()], LEFT_HEEL_COLORS + LEFT_TOE_COLORS + RIGHT_HEEL_COLORS + RIGHT_TOE_COLORS, MARKER_SIZE, LEFT_HEEL_LABELS + LEFT_TOE_LABELS + RIGHT_HEEL_LABELS + RIGHT_TOE_LABELS, True, (60,5))