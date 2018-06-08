import pandas as pd
import numpy as np
import os, os.path


csvPATH = '/home/gryphonlab/Ioannis/Works/RML/Core/deleteSpectrograms.csv'

df = np.array(pd.read_csv(csvPATH))
wrongs = list(df[:,0])
del_counter = 0

for wrong in wrongs:

    index = wrong.find('t_')
    path = wrong[:index]
    time = float(wrong[index+2:-4])

    files = os.listdir(path)
    for f in files:

        file_time = float(f[2:-4])

        if abs(time-file_time)<=0.001:

            os.remove(path+f)
            del_counter += 1
            print( '['+str(del_counter)+'] Deleted file at: '+path+f )