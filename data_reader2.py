# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:14:39 2021

@author: Graysire
"""
import csv
import os
import numpy as np




''' 
returns  all files in directory dir that end in filetype
'''
def get_files(dir, filetype):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if(name.endswith(filetype)):
                r.append(os.path.join(root, name))
    return r

'''
file is a csv file consisting of 1 empty row, 1 row of labels, and rows of numerical data
returns a numpy array containing all of the data
'''
def get_file_data(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(open(file, 'r'))
    
        dataList = []
    
        for row in reader:
            # if(reader.line_num == 2):
            #     dataList.append(row)
            #return only the rows that contain data, convert the strings to floats
            if(reader.line_num > 2):
                try:
                    dataList.append([float(i) for i in row])
                except:
                    print("Error in file:")
                    print(file)
                    print("At line: ")
                    print(reader.line_num)
        
        csvfile.close()
        return np.array(dataList)

'''
Gets all .CSV files from directory dir
Assumes that .CSV files come in sets of 3, PS.CSV, WL.CSV, WR.CSV
Returns a Tuple that consists of a name for the data, PS data, WL data, and WR data
with the data being in numpy arrays
PS data should be shape [n:14] and WL/WR data should be shape [d:5]
'''

def get_all_csv_data(dir):
    fileList = get_files(dir, ".CSV")
    
    dataSetList = []
    
    for i in range(0,len(fileList),3):
        #used to name the tuple, removes the directory and file from the name
        # ex. dir/a/b/PS.CSV becomes "a b"
        tupleName = fileList[i][len(dir) + 1:len(fileList[i]) - 7].replace("\\", "_").replace("_","-",1)
        #format name
        #tupleName.replace("\\", "_")#.replace("_","-",1)
        
        
        dataSetList.append((tupleName, get_file_data(fileList[i]), get_file_data(fileList[i + 1]), get_file_data(fileList[i + 2])))
    print(len(dataSetList))
    return dataSetList

def get_all_data_truncated(dir):
    data = get_all_csv_data(dir)
    data = np.array(data)
    for i in range(1):
        print(type(data[i][2][0]))
        print(data[i][2][:][0])
        
        start = int(min(data[i][2][0][2], data[i][3][0][2]))
        
        data[i][2][:][2] - start
        
        #print(int(min(data[i][2][0][2], data[i][3][0][2])))
        data[i][1] = data[i][1][start:]
    
    return data

# l = get_all_csv_data(os.path.join(os.getcwd(), "WalkEvenData"))

get_all_data_truncated(os.path.join(os.getcwd(), "WalkEvenData"))

