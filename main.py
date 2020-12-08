# -*- coding: utf-8 -*-
"""
last update 8/12/2020
@author: Tonatiuh Hernández-del-Toro (tonahdztoro@gmail.com)

Code for the paper 
"An algorithm for onset detection of linguistic segments in Continuous Electroencephalogram signals"
presented on the 11th MAVEBA

arXiv: 
doi: 10.36253/978-88-6453-961-4
"""

import os
import mat4py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier




def extract_folds(subject, FeatureSet):
    """Returns the train, test, and epoch folds"""
    
    #Set the dir path
    script_dir = os.path.dirname(__file__)
    rel_path = 'DataSets/' + FeatureSet + '/corpusFoldsS' + str(subject + 1) + '.mat'
    file_path = os.path.join(script_dir, rel_path)
    mat = mat4py.loadmat(file_path);

    #Extracts the train, test, and epoch folds
    corpusFolds = mat['corpusFolds'];
    trainCorpusFolds = corpusFolds['trainCorpus'];
    testCorpusFolds = corpusFolds['testCorpus']; 
    testEpochFolds = corpusFolds['testEpoch'];
    return trainCorpusFolds, testCorpusFolds, testEpochFolds



def train_classifier(OnsetAccuracy, trainCorpusFolds, testCorpusFolds, testEpochFolds, k):
    #select data from the folds
    trainCorpus = np.array( trainCorpusFolds[k] );
    np.random.shuffle(trainCorpus);
    testCorpus = np.array( testCorpusFolds[k] );
    testEpoch = np.array( testEpochFolds[k] );
    testEpoch = testEpoch[:,1:3];
    sizeOfCorpus = len(trainCorpus[1,:])-1;
    
    #train and test sets
    X_train = trainCorpus[0:496,0:sizeOfCorpus];
    y_train = trainCorpus[0:496,sizeOfCorpus];
    
    X_test = testCorpus;
    
    #Classifier
    clf = RandomForestClassifier(n_estimators=50,n_jobs=4, random_state=1);
    clf.fit(X_train, y_train);

    
    #make predictions on test corpus
    predictions = clf.predict(X_test);

    #create segments of beginning and ending from the prediction vector
    size = len(predictions);
    onset = np.array([], dtype=int);
    ending = np.array([], dtype=int);
    
    #for avoidinig error when evaluating final + 1 event
    predictions = np.append(predictions, 0);
    
    #find onsets
    i = 0;
    while i < size:
        if predictions[i] == 1:
            onset = np.append(onset, (i)*128+1 );
            j=1;
            while predictions[i+j] == 1:
                j = j+1;
            i = i+j;
            ending = np.append(ending, (i)*128);
        else:
            i += 1;            
    events = np.column_stack((onset, ending));
    
    # Evaluate predictions

    events_predicted = events;
    events_real = testEpoch;
    
    #evaluate onsets
    OnsetTP = 0; 
    
    size_real_onset = range(len(events_real));
    size_predicted_onset = range(len(events_predicted));
    
    for i in size_real_onset:
        real_onset = events_real[i,0];
        lower_limit = real_onset - TETR/2;
        upper_limit = real_onset + TETR/2;
        
        for j in size_predicted_onset:
            predicted_onset = events_predicted[j,0];  
            if lower_limit <= predicted_onset <= upper_limit:
                OnsetTP += 1;
                break;
    
    OnsetTPR = OnsetTP/len(events_real);
    OnsetAccuracy = OnsetAccuracy + OnsetTPR;
    
    return OnsetAccuracy

  
def evaluate_subject(OnsetAccuracies, subject, TETR, FS):
    trainCorpusFolds, testCorpusFolds, testEpochFolds = extract_folds(subject, 'Hurst')
    OnsetAccuracy = 0;
    
    for k in range(5):
        OnsetAccuracy = train_classifier(OnsetAccuracy, trainCorpusFolds, testCorpusFolds, testEpochFolds, k)
        
    OnsetAccuracy = OnsetAccuracy/5;
    OnsetAccuracies = np.append(OnsetAccuracies, OnsetAccuracy);
    
    return OnsetAccuracies
    

def evaluate_datasets(TETR,FS):
    subjects_TPR = np.array([], dtype = float); #average accuracies of each subject
    
    for subject in range(27):
        subjects_TPR = evaluate_subject(subjects_TPR, subject, TETR, FS)
    
    subjects_TPR = np.append(subjects_TPR, np.mean(subjects_TPR))
    subjects_TPR = np.around(subjects_TPR,2)
    
    return subjects_TPR

#%%

TETR = 3*(128)
#TETR = 4*(128)
FS = ['Hurst', 'Stat']


Results1 = evaluate_datasets(TETR,FS[0])
Results2 = evaluate_datasets(TETR,FS[1])



#%%
fig, ax = plt.subplots()
index = np.arange(28)
bar_width = 0.4
opacity = 0.6

rects1 = plt.bar(index, Results1[0], bar_width,
                 alpha=opacity,
                 color='b',
                 label='Feature set 1')

rects2 = plt.bar(index + bar_width, Results2[0], bar_width,
                 alpha=opacity,
                 color='g',
                 label='feature set 2')

plt.xlabel('Subject')
plt.ylabel('TPR')
plt.title('TPR with TETR = 3s')
plt.xticks(index + bar_width,
           ('1','','','','5','','','','','10','','','','','15','','','','','20','','','','','25','','','Mean'))
plt.ylim(0.5, 1.01)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, shadow=True, ncol=1)
plt.tight_layout()
plt.show()











