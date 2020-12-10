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
    """Returns the train, test, and epoch folds from the .mat files"""
    
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



def train_classifier(OnsetTPRs, trainCorpusFolds, testCorpusFolds, testEpochFolds, k):
    """Evaluates the classifier of a given fold and a previous TPR.
    Returns the sum of the previous TPR plus the TPR obtained"""
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
    OnsetTPRs = OnsetTPRs + OnsetTPR;
    
    return OnsetTPRs

  
def evaluate_subject(subject, TETR, FS):
    """Evaluates the true positive rate of a given subject for 5 folds. 
    Returns the TPR of that subject"""
    trainCorpusFolds, testCorpusFolds, testEpochFolds = extract_folds(subject, 'Hurst')
    OnsetTPRs = 0;
    
    for k in range(5):
        OnsetTPRs = train_classifier(OnsetTPRs, trainCorpusFolds, testCorpusFolds, testEpochFolds, k)
        
    OnsetTPR = OnsetTPRs/5;
    
    return OnsetTPR
    

def evaluate_datasets(TETR,FS):
    """Evaluates the whole datasets of each subject given a TETR and a feature set
    Returns the true positive rates for each subject and the mean of all of them"""
    subjects_TPR = np.zeros((28));
    
    for subject in range(27):
        subjects_TPR[subject] = evaluate_subject(subject, TETR, FS)
    
    subjects_TPR[27] = np.mean(subjects_TPR)
    subjects_TPR = np.around(subjects_TPR,2)
    
    return subjects_TPR

def create_figures(Results,TETR_str):
    """Create figures to compare the 2 fueature sets with the given results using a TETR"""
    fig, ax = plt.subplots()
    index = np.arange(28)
    bar_width = 0.4
    opacity = 0.6
    
    plt.bar(index, Results[0], bar_width, alpha=opacity, color='b', label='Feature set 1')
    plt.bar(index + bar_width, Results[1], bar_width, alpha=opacity, color='g', label='Feature set 2')
    
    plt.xlabel('Subject')
    plt.ylabel('TPR')
    plt.title('TPR with TETR =' +TETR_str)
    plt.xticks(index + bar_width,
               ('1','','','','5','','','','','10','','','','','15','','','','','20','','','','','25','','','Mean'))
    plt.ylim(0.5, 1.01)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, shadow=True, ncol=1)
    plt.tight_layout()
    
    #Saves the figure int he figures folder of the main path
    script_dir = os.path.dirname(__file__)
    rel_path = 'figures/' + TETR_str + '.pdf'
    file_path = os.path.join(script_dir, rel_path)
    plt.savefig(file_path, dpi=400, format='pdf', bbox_inches="tight")
    
    plt.show()
    return




if __name__ == '__main__':
    FS = ['Hurst', 'Stat']
    TETR_str = ['3s','4s']
    
    TETR = 3*128
    Results1 = evaluate_datasets(TETR,FS[0])
    Results2 = evaluate_datasets(TETR,FS[1])
    Results = np.vstack([Results1,Results2])
    create_figures(Results,TETR_str[0])
    
    TETR = 4*128
    Results1 = evaluate_datasets(TETR,FS[0])
    Results2 = evaluate_datasets(TETR,FS[1])
    Results = np.vstack([Results1,Results2])
    create_figures(Results,TETR_str[1])


