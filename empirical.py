import pandas as pd
from experiments import *
import numpy as np
import matplotlib.pyplot as plt
import random as random
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as dtc
import datetime

data = pd.read_csv('./Empirical/recidivism.csv')

#check value counts of each variable

data.iloc[:,0].value_counts()
data.iloc[:,1].value_counts()
data.iloc[:,2].value_counts()
data.iloc[:,3].value_counts()
data.iloc[:,4].value_counts()
data.iloc[:,5].value_counts()
data.iloc[:,6].value_counts()
data.iloc[:,7].value_counts()
data.iloc[:,8].value_counts()
data.iloc[:,9].value_counts()
data.iloc[:,10].value_counts()
data.iloc[:,11].value_counts()
data.iloc[:,12].value_counts()
data.iloc[:,12].hist()
data.iloc[:,13].value_counts()
data.iloc[:,14].value_counts()
data.iloc[:,15].value_counts()
data.iloc[:,16].value_counts()

#how many missings?
data.isna().sum()
#drop 3 missing age and sex
data = data.loc[data['Sex'].notnull()]

# so we want to predict variable 10 using variables 0-9
data = data.iloc[:,0:11]
data.isna().sum()
#fill NA in release type
data.loc[data['Release Type'].isnull(),'Release Type'] ='missing'
#fill NA in supervising
data.loc[data['Main Supervising District'].isnull(),'Main Supervising District'] ='missing'

#replace race with either 1 for white or 2 for non-white
data['race'] = 1
nw_mask = data['Race - Ethnicity'] != 'White - Non-Hispanic'
data.loc[nw_mask,'race'] = 2
#replace sex with dummy variable
data.iloc[:,3] = data.iloc[:,3]=='Female'
#create dummies for age
age_dummies = pd.get_dummies(data.iloc[:,4],drop_first=True)
data = pd.concat([data,age_dummies],axis=1)
data = data.drop('Age At Release ',axis=1)
#merge rare values into other category, and then create dummies for convicting offense
counts = data['Convicting Offense Classification'].value_counts()
mask = data['Convicting Offense Classification'].isin(counts[counts <10].index)
data.loc[mask,'Convicting Offense Classification'] = 'Other'
coc_dummies = pd.get_dummies(data['Convicting Offense Classification'],drop_first=True,prefix='coc')
data= data.drop('Convicting Offense Classification',axis=1)
data = pd.concat([data,coc_dummies],axis=1)
# dummies for convicting offense type
data['Convicting Offense Type'].value_counts()
cot_dummies = pd.get_dummies(data['Convicting Offense Type'],prefix='cot')
data = pd.concat([data, cot_dummies],axis=1)
data = data.drop('Convicting Offense Type', axis=1)
#merge rare values into other category and create dummies for convicting offense subtype
counts = data['Convicting Offense Subtype'].value_counts()
mask = data['Convicting Offense Subtype'].isin(counts[counts <10].index)
data.loc[mask,'Convicting Offense Subtype'] = 'Other'
coc_dummies = pd.get_dummies(data['Convicting Offense Subtype'],drop_first=True,prefix='cos')
data = data.drop('Convicting Offense Subtype',axis=1)
data = pd.concat([data,coc_dummies],axis=1)
#release type - merge all paroles, merge all discharge
data['Release Type'].unique()
parole_mask = data['Release Type'].str.contains('Parole')
special_mask = data['Release Type'].str.contains('Special')
discharge_mask = data['Release Type'].str.contains('Discharge')
missing_mask = data['Release Type'].isnull()
data.loc[parole_mask==True]['Release Type'].unique()
data.loc[special_mask==True]['Release Type'].unique()
data.loc[discharge_mask==True]['Release Type'].unique()
data.loc[missing_mask==True]['Release Type'].unique()
data.loc[discharge_mask==True,'Release Type'] = 'Discharge'
data.loc[parole_mask==True,'Release Type'] = 'Parole'
data.loc[special_mask==True,'Release Type'] = 'Special'
data.loc[missing_mask==True,'Release Type'] = 'missing'
data['Release Type'].value_counts()
rt_dummies = pd.get_dummies(data['Release Type'],drop_first=True, prefix='rt')
data = pd.concat([data,rt_dummies],axis=1)
data = data.drop('Release Type',axis=1)
#data
msd_dummies = pd.get_dummies(data['Main Supervising District'],drop_first=True,prefix='msd')
data = pd.concat([data,msd_dummies],axis=1)
data = data.drop('Main Supervising District', axis=1)
#outcome variable
data['recede'] = data['Recidivism - Return to Prison'] == 'Yes'
data = data.drop('Recidivism - Return to Prison', axis=1)

#drop old race value
data = data.drop('Race - Ethnicity', axis=1)


#reorganize data so that it works for our code :(

crucial_columns = data[['race', 'recede']]
crucial_columns['recede'] = crucial_columns['recede'].astype(float)
data = data.drop('race',axis=1)
data = data.drop('recede',axis=1)
ncov = data.shape[1]
data = pd.concat([data,crucial_columns],axis=1)
#split data into true and test
train_mask = np.random.rand(len(data)) < 0.8
data_train = data[train_mask]
data_test = data[~train_mask]
train_data = data_train
test_data = data_test
train_data_rec = train_data[train_data['recede']==1]
train_data_norec = train_data[train_data['recede']==0]
test_data_rec = test_data[test_data['recede']==1]
test_data_norec = test_data[test_data['recede']==0]
balanced_train_rec = train_data_rec
balanced_train_norec = train_data_norec.iloc[0:int(np.floor(len(train_data_norec)/2.0)),:]
balanced_test_rec = test_data_rec
balanced_test_norec = test_data_norec.iloc[0:int(np.floor(len(test_data_norec)/2.0)),:]
balanced_train = pd.concat([balanced_train_rec, balanced_train_norec],axis=0)
balanced_test = pd.concat([balanced_test_rec,balanced_test_norec],axis=0)
def run_iterative_boosting_empirical(ncov,train_data,test_data,val_data_size, max_learners, fairness=True):
    learners = []
    alphas = []
    current_log_weights = []
    frac_blue = []
    val_data = train_data.copy(deep=True)
    validation_data_gen = empirical_data_generator(val_data)
    combined_classifiers = []
    for i in range(max_learners):
        learners, alphas, current_log_weights = boost_iterative(train_data,ncov, validation_data_gen, learners, alphas, current_log_weights, fairness,val_data_size)
        frac_blue.append(np.sum(np.exp(current_log_weights[train_data.iloc[:,ncov]==2]))/np.sum(np.exp(current_log_weights)))
    error_metrics_train = {'learners': [], 'accs':[],'exp':[]}
    error_metrics_test = {'learners': [], 'accs':[],'exp':[]}
    for j in range(max_learners):
        hypothesis = make_combined_hypothesis(learners, alphas, j+1,ncov)
        classifier = make_combined_classifier(hypothesis)
        metrics_train, metrics_test, exp_loss_train, exp_loss_test = get_metrics(hypothesis,classifier, train_data,test_data,ncov)
        error_metrics_train['learners'].append(j+1)
        error_metrics_test['learners'].append(j+1)
        error_metrics_train['accs'].append(metrics_train)
        error_metrics_test['accs'].append(metrics_test)
        error_metrics_train['exp'].append(exp_loss_train)
        error_metrics_test['exp'].append(exp_loss_test)
        combined_classifiers.append(classifier)
    return error_metrics_train, error_metrics_test,frac_blue,alphas,



max_learners = 250
val_data_size=500

results_fair_b = run_iterative_boosting_empirical(ncov,balanced_train,balanced_test,val_data_size, max_learners, fairness=True)
fair_df = get_res_df(results_fair_b)
plot_exp(fair_df, 'fair', 'empirical_balanced_'+str(datetime.datetime.now()))

results_vanilla_b = run_iterative_boosting_empirical(ncov,balanced_train,balanced_test,val_data_size, max_learners, fairness=False)
vanilla_df = get_res_df(results_vanilla_b)
plot_exp(vanilla_df, 'vanilla', 'empirical_balanced_'+str(datetime.datetime.now()))

plot_err_ratios_together('ratio',fair_df['train_err'], vanilla_df['train_err'],'empirical_balanced', 'train')
plot_err_ratios_together('ratio',fair_df['test_err'], vanilla_df['test_err'],'empirical_balanced', 'test')
plot_err_ratios_together('log_exploss_dif',fair_df['train_exp'], vanilla_df['train_exp'],'empirical_balanced', 'train')
plot_err_ratios_together('log_exploss_dif',fair_df['test_exp'], vanilla_df['test_exp'],'empirical_balanced', 'test')



results_fair = run_iterative_boosting_empirical(ncov,data_train,data_test,val_data_size, max_learners, fairness=True)
fair_df = get_res_df(results_fair)
plot_exp(fair_df, 'fair', 'empirical_unbalanced'+str(datetime.datetime.now()))

results_vanilla = run_iterative_boosting_empirical(ncov,data_train,data_test,val_data_size, max_learners, fairness=False)
vanilla_df = get_res_df(results_vanilla)
plot_exp(vanilla_df, 'vanilla', 'empirical_unbalanced'+str(datetime.datetime.now()))

plot_err_ratios_together('ratio',fair_df['train_err'], vanilla_df['train_err'],'empirical_unbalanced', 'train')
plot_err_ratios_together('ratio',fair_df['test_err'], vanilla_df['test_err'],'empirical_unbalanced', 'test')
plot_err_ratios_together('log_exploss_dif',fair_df['train_exp'], vanilla_df['train_exp'],'empirical_unbalanced', 'train')
plot_err_ratios_together('log_exploss_dif',fair_df['test_exp'], vanilla_df['test_exp'],'empirical_unbalanced', 'test')


ax = pd.DataFrame(results_fair_b[3]).plot(legend=False)
ax.set_xlabel("Learners")
ax.set_ylabel('Minority Weight')
plt.savefig('./experiments/fair_balanced_min_weight' + str(datetime.datetime.now()) + '.png')
plt.close()

ax = pd.DataFrame(results_vanilla_b[3]).plot(legend=False)
ax.set_xlabel("Learners")
ax.set_ylabel('Minority Weight')
plt.savefig('./experiments/vanilla_balanced_min_weight' + str(datetime.datetime.now()) + '.png')
plt.close()

ax = pd.DataFrame(results_fair[3]).plot(legend=False)
ax.set_xlabel("Learners")
ax.set_ylabel('Minority Weight')
plt.savefig('./experiments/fair_unbalanced_min_weight' + str(datetime.datetime.now()) + '.png')
plt.close()

ax = pd.DataFrame(results_vanilla[3]).plot(legend=False)
ax.set_xlabel("Learners")
ax.set_ylabel('Minority Weight')
plt.savefig('./experiments/fair_unbalanced_min_weight' + str(datetime.datetime.now()) + '.png')
plt.close()