from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, precision_score, precision_score, recall_score, f1_score, auc, roc_curve
import scipy
import scipy.sparse as sps

def find_gain_lift(proba,y):
    N= len(y)
    A = np.empty([N,2])
    A[:,0]= proba
    A[:,1] = y
    B=pd.DataFrame(A)
    B.columns = ['Target Score','Target']
    B = B.sort_values(['Target Score'], ascending = False)
    B.reset_index(level=None, drop=True, inplace=True)
    B['percentage'] =  100 *(np.arange(0,N)+1)/N  #
    Total_Target = B['Target'].sum()
    B['Gain Target'] =  np.cumsum(B['Target'].values)/Total_Target
    B['Lift Target'] =  B['Gain Target']/B['percentage']
    return B
    
def gain_plot(proba,y,model_name="") :
    table = find_gain_lift(proba,y)
    plt.plot(table['percentage'].values, table['Gain Target'].values,label=model_name+ " Target")  #
    table = find_gain_lift(1-proba,1-y)
    plt.plot(table['percentage'].values, table['Gain Target'].values,label=model_name + " NonTarget")
    plt.plot(table['percentage'].values, 0.01*table['percentage'].values,'y--') #, label="Baseline"
    plt.grid(True)
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.ylabel('Gain')
    plt.xlabel('Percentage of Sample')
    plt.title("Cumulative Gain Curve")
    
def lift_plot(proba,y,model_name="") :
    table = find_gain_lift(proba,y)
    plt.plot(table['percentage'].values, table['Lift Target'].values,label=model_name+ " Target") # 
    table = find_gain_lift(1-proba,1-y)
    plt.plot(table['percentage'].values, table['Lift Target'].values,label=model_name + " NonTarget")
    plt.plot([0, 100], [1, 1], 'y--') #, label='Baseline'
    plt.grid(True)
    plt.xlim([0, 100])
    plt.legend(loc = 'upper right')
    plt.ylabel('Lift')
    plt.xlabel('Percentage of Sample')
    plt.title("Lift Curve")
    

    

def roc_plot(probab, y_train, model_name=""):
    fpr, tpr, thresholds = roc_curve(y_train, probab)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,  label = model_name+' AUC = %0.2f' % roc_auc) 
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'y--')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC Curve")
    
    
def ks_plot(proba_train, y_train, model_name=""):
    N= len(y_train)
    A = np.empty([N,2])
    A[:,0]= proba_train
    A[:,1] = y_train
    B=pd.DataFrame(A)
    B.columns = ['Target Score','Target']
    B = B.sort_values(['Target Score'], ascending = True ) #
    B.reset_index(level=None, drop=True, inplace=True)
    B['NonTarget'] = 1 - B['Target']

    Total_Target = B['Target'].sum()
    Total_nonTarget = B['NonTarget'].sum()

    B['Target Cum Count'] = np.cumsum(B['Target'])
    B['NonTarget Cum Count'] = np.cumsum(B['NonTarget'])
    B['Target Cum Probability'] =  B['Target Cum Count']/Total_Target
    B['NonTarget Cum Probability'] =  B['NonTarget Cum Count']/Total_nonTarget
    B['K-S statistic'] = B['NonTarget Cum Probability'] - B['Target Cum Probability']
    #display(B.sort_values(['K-S statistic'], ascending = False).head())
    
    idx_max = B['K-S statistic'].idxmax()
    ks =B['K-S statistic'].iloc[idx_max]
    p= B['Target Score'].iloc[idx_max]
    x1, x2, y1, y2 = p, p, B['Target Cum Probability'].iloc[idx_max], B['NonTarget Cum Probability'].iloc[idx_max]
    
    plt.plot(B['Target Score'].values, B['Target Cum Probability'].values, label = "Target")
    plt.plot(B['Target Score'].values, B['NonTarget Cum Probability'].values,'r', label = "Non-Target")
    plt.plot((x1, x2), (y1, y2), 'g--',label = model_name+ ' T-S = %0.3f' % ks + " at p = %0.3f " %p)  #
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Cut Off')
    plt.ylabel('Cumulative Probability')
    plt.title("Kolmogorov-Smirnov Plot")
    plt.grid()
    
    
def ks_chart(proba_train, y_train):
    data = pd.DataFrame(proba_train.reshape(-1,1))
    data.columns = ['Target Score']
    data['Target']=y_train.reshape(-1,1)
    data['Target'] = data['Target'].apply(int)
    data['NonTarget'] = (1 - data['Target']).apply(int)

    Total_Target = data['Target'].sum()
    Total_nonTarget = data['NonTarget'].sum()

    data['Decile'] = pd.qcut(data['Target Score'], 10, labels = False)
    Ranges =  [np.round([a.left,a.right],3) for a in pd.qcut(proba_train, 10).categories.tolist()]

    grouped = data[['Target','NonTarget','Decile']].groupby('Decile').sum()  #, as_index = False
    grouped = grouped[['Target','NonTarget']].reset_index(drop=True)
    grouped['Decile']=np.arange(10,0,-1) 

    grouped['Record Count'] = (grouped['Target']+grouped['NonTarget']).apply(int)
    grouped['Probabilty Ranges'] = Ranges
    grouped['% Targets'] = 100 * grouped['Target'] / Total_Target
    grouped['% NonTargets'] = 100 * grouped['NonTarget'] / Total_nonTarget

    grouped = grouped.sort_values('Decile',ascending=True)
    grouped['% Target Cum']= np.cumsum(grouped['% Targets'].values)
    grouped['% NonTarget Cum']= np.cumsum(grouped['% NonTargets'].values)
    grouped['K-S']=grouped['% Target Cum'] - grouped['% NonTarget Cum'] 
    flag = lambda x: '<----' if x == grouped['K-S'].max() else ''
    grouped['Max K-S'] = grouped['K-S'].apply(flag)

    grouped=grouped[['Decile','Probabilty Ranges','Record Count','Target','NonTarget','% Targets','% NonTargets','% Target Cum','% NonTarget Cum','K-S','Max K-S']]
    return grouped.round(3)



def TableLog(proba,y_values):
    a = pd.DataFrame(pd.qcut(proba, 10, labels=False))
    a.columns = ['deciles']
    b= pd.DataFrame(pd.qcut(proba, 10).categories)
    qcuts = pd.DataFrame(a.groupby(['deciles']).size())
    b.reindex(qcuts.index)
    b.columns = ['ranges']
    A=pd.concat([qcuts,b], axis =1 )
    A.columns = ['record_count','range']

    temp = pd.DataFrame(y_values)
    temp.columns = ['Labels']
    temp['deciles']= a
    FCs = temp[['Labels','deciles']].groupby('deciles').sum()
    FCs.columns = ['Target_Sum']
    FCs = FCs.sort_index(ascending=False)
    FCs['Percentage_Of_Response']=   100*np.cumsum(FCs['Target_Sum'].values)/np.sum(FCs['Target_Sum'].values)
    FCs = FCs.sort_index(ascending=True)
    table = pd.concat([A,FCs], axis =1)
    return table


def show_metrics(proba ,y_train):
    predictions = (proba>=0.5)+0
    Confusion_matrix = confusion_matrix(y_train, predictions)
    sensitivity = Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[0,1])
    specificity = Confusion_matrix[1,1]/(Confusion_matrix[1,0]+Confusion_matrix[1,1])
    Precision_score = precision_score(y_true=y_train, y_pred=predictions)
    Recall_score = recall_score(y_true=y_train, y_pred=predictions)
    F1_score = f1_score(y_true=y_train, y_pred=predictions)
    fpr, tpr, _ = roc_curve(y_train, proba)
    Auc = auc(fpr, tpr)
    gini = gini_normalized(y_train, proba)

    summary = [Confusion_matrix,Precision_score,Recall_score,sensitivity,specificity,F1_score,Auc,gini]
    summary = pd.DataFrame(summary)
    summary[1] = ['[tn, fp],[fn, tp]','precision','recall','sensitivity','specificity','f1','auc','gini_normalized'] 
    summary.columns = ['Value','Criteria']
    return summary






def gini_normalized(actual, pred):
    #https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation
    def gini(actual, pred):
        assert (len(actual) == len(pred))
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses

        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
    return gini(actual, pred) / gini(actual, actual)


    

def logit_model(X,y,Predictor_names,class_weight='not-balanced'):    
# Descrption:

#Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    if class_weight=='balanced':
        logistic = LogisticRegression(C=10**6,verbose = 0, fit_intercept=True, class_weight="balanced")
    if class_weight=='not-balanced':
        logistic = LogisticRegression(C=10**6,verbose = 0, fit_intercept=True)
   
    logistic.fit(X,y);

    # Calculate matrix of predicted class probabilities. 
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    predProbs = np.matrix(logistic.predict_proba(X))
    # Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack([np.ones([X.shape[0],1]), X])
    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    
    #V = np.matrix(np.zeros(shape = (X_design.shape[0], X_design.shape[0])))
    #np.fill_diagonal(V, np.multiply(predProbs[:,0], predProbs[:,1]).A1)
    # Covariance matrix
    #covLogit = np.linalg.inv(X_design.T * V * X_design)
    temp =np.multiply(predProbs[:,0], predProbs[:,1]).A1
    V = sps.diags(temp)
    covLogit = np.linalg.pinv(X_design.T.dot(V.dot(X_design)))  #pinv
    
    # Standard errors
    se  = np.sqrt(np.diag(covLogit))
    coef = np.append(logistic.intercept_, logistic.coef_)
    zscore = coef / se

    #It tests the null hypothesis that the associated coefficient value is 0
    # The smaller the p-value, the more likely beta not  equal 0.
    #p-values less than 0.05 indicate that the predictor significantly contributes to the model. 
    #Values greater than 0.05 indicate that the null hypothesis has not been rejected and that the predictor
    #does not contribute to the model.
    pvalue =2*(1- scipy.stats.norm.cdf(abs(zscore)))
    loci = coef - 1.96 * se
    upci = coef + 1.96 * se

    # statistics of odd_ratio
    # odd = mu / (1-mu)
    #The odds ratio of x = = 1 is the ratio of the odds of x = 1 to the odds of x = 0.
    #If the odds ratio for age is 1.01 and response is died, we can assert that the odds
    #of death is 1% greater for each 1 year greater age of a patient.
    o_r = np.exp(coef) # odds ratios
    delta = o_r*se

    Results = pd.DataFrame([coef, se, zscore, pvalue, loci, upci,o_r,delta]).transpose()
    Results.columns = ['coef', 'se', 'zscore', 'pvalue', 'loci', 'upci','odd_ratio','delta']
    names =  ["Intercept"] + list(Predictor_names)  # 
    Results.index = names

    Results['Code']= ' '
    for i in range(0,Results.shape[0]):
        if 0<=Results['pvalue'].iloc[i]<=0.001:
            Results['Code'].iloc[i] = '***'
        if 0.001<Results['pvalue'].iloc[i]<=0.01:
            Results['Code'].iloc[i] = '**' 
        if 0.01<Results['pvalue'].iloc[i]<=0.05:
            Results['Code'].iloc[i] = '*'
        if 0.05<Results['pvalue'].iloc[i]<=0.1:
            Results['Code'].iloc[i] = '.' 
            
    #display(Results.round(2)) 
    return logistic,Results


#def gain_plot(probas, FCs,model_name=""): # , df_train
    #npos = np.sum(FCs)
    #n = len(FCs)
    #index = np.argsort(probas) 
    #index = index[:: -1] 
    #sort_probabs = FCs[index]
    #cpos = np.cumsum (sort_probabs)
    #rappel = cpos/npos
    #taille= np.arange(start=1,stop=n+1,step=1) 
    #taille = taille/n
    #plt.plot(taille,rappel,  label = model_name)
    #plt.plot (taille,taille,'r--') 
    #plt.grid(True)
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.legend(loc = 'lower right')
    #plt.ylabel('Gain')
    #plt.xlabel('Percentage of Sample')
    #plt.title("Cumulative Gain Curve")
