#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

poi_label=['poi']
financial_features=['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_feature=['to_messages','from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']


mfeatures_list=poi_label+financial_features+email_feature

### this is used for decision tree, final algorithm
features_list=['poi','salary','bonus','total_payments','long_term_incentive','expenses']


#k means
#features_list=['poi','salary','bonus','total_payments','long_term_incentive','expenses']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

    
    
###Data exploration - START
"""
As part of data exploration strategy I have done the following Analysis:
    1. Find the number of data points.
    2. POI/Non POI in dataset.
    3. How many features have NaN and their count as well as the percentage.
    4. I tried using many combination of features and finally settled with few of them. I tried to change the 
    number of features and observed the performance of the algorithm. Finally chose those features which
    gave better results. I have provided my comments which speaks about the features used for various algo.

"""


""" Find total number of data points"""
print ("Total number of data points:: ",len(data_dict))


print ("Printing sample data")
for key,value in data_dict.items():
    for k,v in value.items():
        print (k)
    break


""" Find allocation across classes (POI/non POI)"""
poi_counter=0
for key,value in data_dict.items():
    if value["poi"]== True:
        poi_counter+=1
        
print ("Total number of poi in dataset:: ",poi_counter) 

non_poi_counter=0
for key,value in data_dict.items():
    if value["poi"]== False:
        non_poi_counter+=1
        
print ("Total number of NON - poi in dataset:: ",non_poi_counter) 

# To find number of feature for each person
"""
for key,value in data_dict.items():
    print (key,len(value))
"""   
    
###This function will find NaN in the features and returns the percentage(whole data dict is considered)
def findpercentage(param1):
    count_perc=0
    for k,v in data_dict.items():
        if v[param1]=="NaN":
            count_perc+=1
    print ("Count of NaN in the follwing feature"+":::"+param1,count_perc)
    #print ("Percentage using function",(float(count_perc)/float(len(data_dict)))*100)
    percent=(float(count_perc)/float(len(data_dict)))*100
    return percent
            

print ("Percentage of NaN of feature total payment in whole data dict :: ",findpercentage("total_payments"))
print ("Percentage of NaN of feature salary in whole data dict :: ",findpercentage("salary"))
print ("Percentage of NaN of feature bonus in whole data dict :: ",findpercentage("bonus"))
print ("to_messages::",findpercentage("to_messages"))
print ("deferral_payments::",findpercentage("deferral_payments"))
print ("exercised_stock_options::",findpercentage("exercised_stock_options"))
print ("restricted_stock::",findpercentage("restricted_stock"))
print ("shared_receipt_with_poi::",findpercentage("shared_receipt_with_poi"))
print ("restricted_stock_deferred::",findpercentage("restricted_stock_deferred"))
print ("total_stock_value::",findpercentage("total_stock_value"))
print ("expenses::",findpercentage("expenses"))
print ("loan_advances::",findpercentage("loan_advances"))
print ("from_messages::",findpercentage("from_messages"))
print ("other::",findpercentage("other"))
print ("from_this_person_to_poi::",findpercentage("from_this_person_to_poi"))
print ("poi::",findpercentage("poi"))
print ("director_fees::",findpercentage("director_fees"))
print ("deferred_income::",findpercentage("deferred_income"))
print ("long_term_incentive::",findpercentage("long_term_incentive"))
print ("email_address::",findpercentage("email_address"))
print ("from_poi_to_this_person::",findpercentage("from_poi_to_this_person"))

"""loan_advances,director_fees,restricted_stock_deferred and deferral payments have large number of missing vlues - NaN in the data dict. 
Also bonus has 64 NaN which is 43.8% of whole data dict. As per my intuition the features which 
has more missing values are not important. But the bonus feature looks like an important feature. """




#Count POIs in the E+F dataset have NaN for their total payments  
def findpoiNan(param1):
    ct_nan_poi=0
    for k,v in data_dict.items():
        if v["poi"]==True:
            if v[param1]=="NaN":
                ct_nan_poi+=1
    print ("Count POIs in the E+F dataset have NaN for:: "+param1+"::",ct_nan_poi)
                
findpoiNan("total_payments")
findpoiNan("salary")
findpoiNan("bonus")
###  DATA exploration --  END 


### Task 2: Remove outliers -- START

#Write a function to visualize the features and check for outliers.
def plottwoFeatures(feature_x,feature_y,x_label,y_label):
    features_plot = [feature_x,feature_y]
    data = featureFormat(data_dict, features_plot)
    for point in data:
        feature_x = point[0]
        feature_y = point[1]
        plt.scatter(feature_x,feature_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
  
plottwoFeatures("salary","bonus","Salary","Bonus")


""" Find the biggest enron outlier """
""" Below code will put the name and salary in dictionary tmp_dict """
tmp_dict={}
for key,value in data_dict.items():
    if isinstance(value["salary"],int):
        name=key
        v_sal=value["salary"]
        tmp_dict[name]=v_sal

        

"""sort the list by values/salary """
sort_tmp_dict=sorted(tmp_dict.values())
length_of_dict=len(sort_tmp_dict)
print ("length of dictionary",length_of_dict)
val_highest=sort_tmp_dict[length_of_dict - 1]
print ("Highest value/salary :: ",val_highest)

biggest_outlier=""
for k,v in tmp_dict.items():
    if v==val_highest:
        print ("Biggest outlier",k)
        biggest_outlier=k

""" below code will remove the outlier"""
data_dict.pop(biggest_outlier,0)
print ("Biggest outlier is removed")


"""
In this section I found that TOTAL is the biggest Enron outlier and I have removed it from the data set.
I could not find any other value which should be removed. TOTAL is removed since its an outlier and not a 
name of person.
"""

###  Remove outliers -- END


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset={}
my_dataset = data_dict

"""
print ("Printing sample data")
for key,value in my_dataset.items():
    for k,v in value.items():
        print (k)
    break
"""    


def computeFractionMessages(poi_message,all_message):
    if isinstance(poi_message,int) and isinstance(all_message,int):
        fraction=float(poi_message)/float(all_message)
    
    if poi_message=="NaN" or all_message=="Nan":
        fraction=0
        
    return fraction
    
print ("##############################################################################################")
for name in my_dataset:
    data_point=my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFractionMessages(from_poi_to_this_person,to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFractionMessages(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi


# Following code is just to see first item in dataset.
for k,v in my_dataset.items():
    print (k,v)
    break


#Will add the newly created feature in feature list
new_feature=['fraction_from_poi','fraction_to_poi']
my_feature_list=mfeatures_list + new_feature

def featureFormatAndSplit(dataset,algofeatureList):
    data = featureFormat(dataset, algofeatureList, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return labels,features
   
# We will comment the line below because it is used during PCA. my_feature_list includes features added/created/transformed
#by user and we have measured the performance of it.
#labels,features=featureFormatAndSplit(my_dataset,my_feature_list)

labels,features=featureFormatAndSplit(my_dataset,features_list)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3, random_state=42)


######################  Function to find True positive/ False negative  ############################

#find true positive. In this case, we define a true positive as a case where both the actual 
#label and the predicted label are 1
def findTruePositive(predictionList,labelTest):
    true_positive_counter=0
    for pd,lb in zip(predictionList,labelTest):
        if pd==1:
            if pd==lb:
                true_positive_counter+=1
    return true_positive_counter


#FInd false negative
def findFalseNegative(predictionList,labelTest):
    false_neg_counter=0
    for pd,lb in zip(predictionList,labelTest):
        if pd==0 and lb==1:
            false_neg_counter+=1
            
    return false_neg_counter
    
#######################    Function END  ###################################################

############################   MinMaxScaler  START##############################################
"""
Min Max Scaler is used for feature scaling.

"""

scaler=MinMaxScaler()
r_features_train=scaler.fit_transform(features_train)
r_features_test=scaler.transform(features_test) 
    
#############################  MinMaxScaler   END ############################################## 


######################################## StandardScaler  START#########################################

"""
This is also part of feature Scaling.
"""
sd_scaler=StandardScaler()
scaler_features_train=sd_scaler.fit_transform(features_train)
scaler_features_test=sd_scaler.transform(features_test)

########################################   Standard Scaler END     ################################### 

################################ SelectKBest ######################################################

selector=SelectKBest(k=5)
X_new=selector.fit_transform(features_train,labels_train)
ftt_test=selector.transform(features_test)
print ("Scores using SelectKBest:: ",selector.scores_)

###########################   SelectKBest END  ######################################################




#######    PCA/pipeline with Decision tree   ###############################################################
estimators = [('reduce_dim', PCA(n_components=5,copy=True)), ('DT',DecisionTreeClassifier(criterion='gini', 
              splitter='best', max_depth=8, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                             random_state=42, max_leaf_nodes=None, 
                             class_weight=None, presort=False))]
clf_pipe_DT = Pipeline(estimators)
clf_pipe_DT.steps[0]
clf_pipe_DT.named_steps['reduce_dim']
clf_pipe_DT.fit(features_train,labels_train)
pred_pipeline=clf_pipe_DT.predict(features_test) 
pred_pipeline_python=pred_pipeline.tolist()
print ("Precision score using pipeline::",precision_score(labels_test, pred_pipeline_python))
print ("Recall score using pipeline::",recall_score(labels_test, pred_pipeline_python))  
print ("TRUE POSITIVE for pipeline ",findTruePositive(pred_pipeline_python,labels_test))
print ("FALSE NEGATIVE  for pipeline ",findFalseNegative(pred_pipeline_python,labels_test))

###########################################################################################################


############################  Decision Tree START  #############################################

###  As part of Final Analysis we will use Decision Tree.

    
clf_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=6, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                             random_state=42, max_leaf_nodes=None, 
                             class_weight=None, presort=False)
clf_DT=clf_DT.fit(r_features_train,labels_train)
pred=clf_DT.predict(r_features_test,labels_test)

#print ("prediction of decision tree",pred)
    
###How many POIs are predicted for the test set for your POI identifier?
###(Note that we said test set! We are not looking for the number of POIs in the whole dataset.)
poi_counter=0
for x in np.nditer(pred):
    if x==1.0:
        poi_counter+=1
    

print ("POIs predicted for the test set:: ",poi_counter)
print ("Total people in test set",pred.size) # Numpy array

#converting numpy array to pythton list
pred_python_list=pred.tolist()
print ("pred python list:: ",pred_python_list)
print ("length of pred_python_list:: ",len(pred_python_list))
print ("length of labels test:: ",len(labels_test))

        
list_imp_features=clf_DT.feature_importances_
print ("Length of list  of list_imp_features:: ",len(list_imp_features))

#to find the most important/powerful feature 
numb_ctk=0
for k in list_imp_features:
    if k > 0.001:
        print ("feature importance",k)
        print ("number of this feature",numb_ctk)
        
    numb_ctk+=1



            

print ("TRUE POSITIVE USING FUNCTION",findTruePositive(pred_python_list,labels_test))
print ("FALSE NEGATIVE  USING FUNCTION",findFalseNegative(pred_python_list,labels_test))
print ("Precision score::",precision_score(labels_test, pred_python_list))
print ("Recall score ::",recall_score(labels_test, pred_python_list))  
print ("Accuracy score :: ",accuracy_score(pred,labels_test)) 


##############################   Decision Tree  END  ###############################################

#######################################   Naive Bayes Gaussian START   ##############################################################
#Naive Bayes

from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(r_features_train,labels_train)
pred_nb=clf_nb.predict(r_features_test)
pred_nb_list=pred_nb.tolist()
print ("Precision score using Naive Bayes::",precision_score(labels_test, pred_nb_list))
print ("Recall score usinf Naive Bayes::",recall_score(labels_test, pred_nb_list))  
print ("TRUE POSITIVE for Naive Bayes USING FUNCTION",findTruePositive(pred_nb_list,labels_test))
print ("FALSE NEGATIVE  for Naive Bayes USING FUNCTION",findFalseNegative(pred_nb_list,labels_test))


# Follwing o/p is obtained with tester.py
"""
        Accuracy: 0.84207       Precision: 0.35819      Recall: 0.23300 F1: 0.28234     F2: 0.25051
        Total predictions: 15000        True positives:  466    False positives:  835   False negatives: 1534   True negatives: 12165

"""

#####################################################################################################    

##################### K means ###########################################
clf_kmeans = KMeans(n_clusters=2,init='k-means++', n_init=5, max_iter=300, tol=0.0001, precompute_distances='auto',
             verbose=0, random_state=42, copy_x=True, n_jobs=1).fit(X_new)
pred_kmeans=clf_kmeans.predict(ftt_test)
pred_python_kmeans=pred_kmeans.tolist()
print ("pred_kmeans:: ",pred_kmeans)
print ("Precision score using kmeans::",precision_score(labels_test, pred_python_kmeans))
print ("Recall score using kmeans::",recall_score(labels_test, pred_python_kmeans))  
print ("TRUE POSITIVE for kmeans USING FUNCTION",findTruePositive(pred_python_kmeans,labels_test))
print ("FALSE NEGATIVE  for kmeans USING FUNCTION",findFalseNegative(pred_python_kmeans,labels_test))

"""
O/P when run from tester.py
Accuracy: 0.83827       Precision: 0.23441      Recall: 0.09400 F1: 0.13419     F2: 0.10679
        Total predictions: 15000        True positives:  188    False positives:  614   False negatives: 1812   True negatives: 12386


"""


clf=clf_DT

dump_classifier_and_data(clf, my_dataset, features_list)