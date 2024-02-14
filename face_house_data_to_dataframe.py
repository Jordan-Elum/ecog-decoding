# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:25:53 2022

@author: Jordan Elum

subjects in a clinical settings (with ECoG implants) are passively shown faces and house during the first experiment (dat1). 
Then in the second experiment in the same subjects (dat2), noise is added to face and houses images and the subject has to detect the faces by pressing a key. 
Two of the subjects don't have keypresses.

This script loads the ECoG dataset for 7 subjects, prints basic electrode info, and plots broadband power 
across electrodes by stimulus type for individual subjects.

adapted from: https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/ECoG/load_ECoG_faceshouses.ipynb

"""
#%% 
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from numpy import load
import pandas as pd
from scipy import signal

#%%  Data retrieval
import os, requests
#replace fname with the absolute path to the file location where the faceshouses.npz file should be stored (i.e. '/data/faceshouses.npz')
fname = r"\\128.95.12.242\Lab_Common\Jordan\code\python\public_github_repo\ecog decoding\528_group_project\data\faceshouses.npz"
url = "https://osf.io/argh7/download"
if not os.path.isfile(fname):
  try:
    r = requests.get(url)
  except requests.ConnectionError:
    print("!!! Failed to download data !!!")
  else:
    if r.status_code != requests.codes.ok:
      print("!!! Failed to download data !!!")
    else:
      with open(fname, "wb") as fid:
        fid.write(r.content)
        
#%% read in data and split up by experiment type
# fname = input('Enter the absolute path of the faceshouses.npz file (no quotes): ')
alldat = np.load(fname, allow_pickle=True)['dat']
save_path = input('Enter the absolute path of the directory in which the output data should be saved (no quotes): ')

#%% split data by experiment (passive task v detection task)
task_passive = []
task_detect = []
for subject in alldat:
    task_passive.append(subject[0])
    task_detect.append(subject[1])

#%%# get broadband power in time-varying windows, add these new 'V_broad_' keys after the V_trace col
i = 0 
for s in task_passive:
    s['subject'] = i
    V = s['V'].astype('float32')
    s['V_broad'] = V
    nt, nchan = V.shape
    nstim = len(s['t_on'])
    trange = np.arange(-200, 400)
    ts = s['t_on'][:, np.newaxis] + trange
    V_epochs = np.reshape(V[ts, :], (nstim, 600, nchan))
    V_house = (V_epochs[s['stim_id'] <= 50]).mean(0)
    V_face = (V_epochs[s['stim_id'] > 50]).mean(0)
    s['V_broad_house_windows'] = V_house
    s['V_broad_face_windows'] = V_face
    s['V_broad_house_windows_all'] = V_epochs[s['stim_id'] <= 50]
    s['V_broad_face_windows_all'] = V_epochs[s['stim_id'] > 50]
    s['V_broad_windows_all'] = V_epochs
    i += 1

#%%  let's find the electrodes that distinguish faces from houses for each subject
for s in task_passive[0:1]:
    plt.figure(figsize=(20, 15))
    for j in range(len(s['locs'])):
      ax = plt.subplot(6, 10, j+1)
      ax.spines['top'].set_color('none')
      ax.spines['right'].set_color('none')
      plt.plot(trange, s['V_broad_house_windows'][:, j],label = 'house')
      plt.plot(trange, s['V_broad_face_windows'][:, j],label = 'face')
      plt.legend(fontsize=11)
      plt.xlabel('Time (ms)')
      plt.ylabel('Broadband (V)')
      plt.title(' Subj: '+ str(s['subject']) + ' Ch '+ str(j))
      plt.xticks([-200, 0, 200])
      plt.ylim([-1,2])
      plt.tight_layout()
      plt.margins(x=0)
      plt.savefig(save_path  + '\\' +  'subject_' + str(s['subject']) +  '_electrode_broadB_means.png')
    plt.show()

#%% make dataframes for each subject with full session V broadband signal by electrode with extra info columns and save in npy and csv formats
for s in task_passive:
    subject = str(s['subject'])
    print('processing data for subject: ', subject)
    df = pd.DataFrame(s['V_broad'])
    df['label'] = 2
    #get index of stim t on times and label matching label column with stim type if not rest 
    for i,j in zip(s['t_on'],s['stim_id']):
        # print(i,j)
        if j <= 50:
            df.loc[i:i+399,'label'] = 0
        else:
            df.loc[i:i+399,'label'] = 1
    np.save(save_path + '\\'  + 'subject_' + subject  + '_passive_task_broadb_full_df.npy', df)  #save as npy format to preserve all col types
    df.head(2).to_csv(save_path + '\\' + 'subject_' +  subject +  '_passive_task_broadb_full_df.csv') #after running above cells to add all sessions, can save entire dataframe as csv to easily reimport quickly for exploration 
    print('dataframe saved as: ', '\n', save_path + 'subject_' + subject +'_passive_task_broadb_full_df')

