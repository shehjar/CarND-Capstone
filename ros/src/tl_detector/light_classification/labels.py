# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:22:54 2017

@author: kauls
"""

import os, glob, csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 

curdir = os.getcwd()
pardir = os.path.abspath(os.path.join(curdir, os.pardir))
targetdir = os.path.join(pardir,'pictures')
files = glob.glob(os.path.join(targetdir,'*.jpg'))

#Defining a dictionary of data
data=[]
# Read images
for i in range(len(files)):
    file = files[i]
    img = mpimg.imread(file)
    plt.imshow(img)
    plt.show()
    print('Picture index= '+str(i))
    usr_inp = input('What label is this?')
    data.append([file, usr_inp])
    #if i >=10:
    #    break

# Write csv 
csvfilename = os.path.join(targetdir, 'data.csv')
with open(csvfilename, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=',')
    for dataline in data:
        datawriter.writerow(dataline)