from scipy.misc import imread
import numpy as np
import pandas as pd
import os
root = './Train' # or './test'. This specifies the directory to select

# go through each directory in the root folder given above
for directory, subdirectories, files in os.walk(root): #https://docs.python.org/3/library/os.html it yields a 3-tuple (dirpath, dirnames, filenames).
    # go through each file in that directory
    for file in files:	
        # read the image file and extract its pixels
        img = imread(os.path.join(directory,file))
        value = img.flatten()
        # the string following the last underscore was inserted into the first column of the dataset in csv because each character
        #followed by two underscore in each diectory listing has been taken as class label.
        value = np.hstack((directory.split("_")[-1],value))
        df = pd.DataFrame(value).T
        df = df.sample(frac=1) # shuffle the dataset
        with open('train.csv', 'a') as dataset: 
            df.to_csv(dataset, header=False, index=False)
