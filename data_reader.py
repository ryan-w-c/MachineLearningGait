# Ryan Cavanagh
import os
import numpy as np
import pandas as pd 
import glob

allTimestamps = []

directory = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), '**'), '**'), '**'), '**')
dataSetList = []
for dir in glob.glob(directory):
    try:
        data = []
        x = dir.split(os.sep)
        data.append(x[-3] + " " + x[-2] + " " + x[-1])

        df = pd.read_csv(os.path.join(dir, "WL.CSV"), skiprows=1)
        x = np.transpose(df.iloc[:,2:4].to_numpy())
        # print(x)
        data.append(x)
        start = x[0][0]
        end = x[1][-1]
        # print(start, end)

        df = pd.read_csv(os.path.join(dir, "WR.CSV"), skiprows=1)
        x = np.transpose(df.iloc[:,2:4].to_numpy())
        # print(x)
        data.append(x)
        start = min(start, x[0][0])
        end = max(end, x[1][-1])
        # print(start, end)

        df = pd.read_csv(os.path.join(dir, "PS.CSV"), skiprows=1)
        x = np.transpose(df.iloc[:,2:14][start:end + 1].to_numpy())
        allTimestamps.append(np.transpose(df.iloc[:,0:1][start:end + 1].to_numpy()))
        # print(x.shape)
        data.append(x)
        # print(data)
        # data.append(df.iloc[:,2:4])
        dataSetList.append(data)
    except:
        print(dir)

# print dataset
for e in dataSetList:
    print("\n\n" + e[0] + "\n\n")
    print("WL:\n", e[1], "\n\nWR:\n",e[2], "\n\nPS:\n", e[3])
    print(e[1].shape)
    print(e[2].shape)
    print(e[3].shape)
    print("\n\n\n")
    break

def get_timestamps(dataIndex):
    return allTimestamps[dataIndex]