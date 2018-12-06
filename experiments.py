import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as random
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as dtc
import datetime
#test_data_freshness='fresh'
test_data_freshness='reused'


def get_prob(beta,features):
    score = float( np.dot(features,beta) / 1.0/np.linalg.norm(features,axis=1))
    prob = 1/(1.0+np.exp(score))
    return prob
def get_label(beta, features):
    prob = get_prob(beta,features)
    label = (1 if (prob >.5) else 0)
    return label

def get_probs_batch(beta,all_features):
    scores = np.dot(all_features,beta) /1.0/np.linalg.norm(all_features,axis=1)
    probs = np.sign(scores)
    #probs = 1/(1.0+np.exp(-scores))
    return probs
def get_label_batch(beta,mean_vector, all_features):
    threshold = np.dot(beta, mean_vector)
    dots = np.dot(all_features,beta)
    #probs = get_probs_batch(beta, all_features)
    labels = dots > threshold
    return labels
def gen_features(ncov,mean_vector,sigma):
    return(sigma*np.random.randn(ncov)+mean_vector)


def gen_features_batch(ncov,ndata, mean_vector, sigma_vector):
    norm_features = np.random.randn(ndata,ncov)
    return np.array(sigma_vector)* norm_features +np.array(mean_vector)


def gen_data_dm(prob_red,ndata,ncov,mu_red,mu_blue,sigma_red, sigma_blue,beta_red, beta_blue):
    colors = np.random.choice([1,2], ndata, p=[prob_red, 1.0-prob_red])
    data = pd.DataFrame(np.zeros((ndata, ncov+2)))
    data.iloc[:,ncov] = colors

    for i in range(ndata):
        if data.iloc[i,ncov]==1:
            features = gen_features(ncov, mu_red, sigma_red)
            label = get_label(beta_red,features)
        else:
            features = gen_features(ncov, mu_blue, sigma_blue)
            label = get_label(beta_blue,features)
        data.iloc[i,0:ncov] =features
        data.iloc[i,ncov+1]=label
    return data

def gen_data_dm_params(prob_red,ndata, ncov,params):
    mu_blue = params['mu_blue']
    mu_red = params['mu_red']
    sigma_red = params['sigma_red']
    sigma_blue = params['sigma_blue']
    beta_blue = params['beta_blue']
    beta_red = params['beta_red']
    colors = np.random.choice([1,2], ndata, p=[prob_red, 1.0-prob_red])
    n_red = sum(colors==1)
    n_blue = sum(colors==2)
    #data = pd.DataFrame(np.zeros((ndata, ncov+2)))
    #data.iloc[:,ncov] = colors
    features_red = gen_features_batch(ncov,n_red,mu_red,sigma_red)
    features_blue = gen_features_batch(ncov,n_blue, mu_blue, sigma_blue)
    labels_red = get_label_batch(beta_red,mu_red,features_red)
    labels_blue = get_label_batch(beta_blue,mu_blue, features_blue)
    red_data = pd.concat([pd.DataFrame(features_red), pd.DataFrame(np.ones(n_red)),  pd.DataFrame(labels_red)], axis=1)
    blue_data = pd.concat([pd.DataFrame(features_blue),pd.DataFrame(2.0*np.ones(n_blue)), pd.DataFrame(labels_blue)],axis=1)
    data = pd.concat([red_data, blue_data],axis=0)
    # for i in range(ndata):
    #     if data.iloc[i,ncov]==1:
    #         features = gen_features(ncov, mu_red, sigma_red)
    #         label = get_label(beta_red,features)
    #     else:
    #         features = gen_features(ncov, mu_blue, sigma_blue)
    #         label = get_label(beta_blue,features)
    #     data.iloc[i,0:ncov] =features
    #     data.iloc[i,ncov+1]=label
    return data


def run_experiment_gnostic_logistic_(ncov,p_red,total_training,test_size,exp_len):
    mu_red = np.random.rand(ncov)
    mu_blue = np.random.rand(ncov)
    sigma_red = .3
    sigma_blue = .3
    red_beta = np.random.randn(ncov)
    blue_beta = np.random.randn(ncov)
    train_data = gen_data_dm(p_red, total_training, ncov, mu_red,mu_blue, sigma_red,sigma_blue,red_beta,blue_beta)
    t=50
    training_errors = pd.Series()
    test_errors = pd.Series()
    red_val_errors = pd.Series()
    blue_val_errors = pd.Series()
    test_err_difs = pd.Series()
    times = pd.Series()
    while t<exp_len:
        print t
        print(datetime.datetime.now())
        test_data = gen_data_dm(p_red, test_size, ncov, mu_red,mu_blue, sigma_red,sigma_blue,red_beta,blue_beta)
        test_data_red = test_data[test_data.iloc[:,ncov]==1]
        test_data_blue = test_data[test_data.iloc[:,ncov]==2]
        #test_data_red = gen_data_dm(1, test_size, ncov, mu_red,mu_blue, sigma_red,sigma_blue,red_beta,blue_beta)
        #test_data_blue = gen_data_dm(0, test_size, ncov, mu_red,mu_blue, sigma_red,sigma_blue,red_beta,blue_beta)
        model = LogisticRegression(random_state=0, solver='lbfgs').fit(train_data.iloc[0:t,0:ncov],train_data.iloc[0:t,(ncov+1)])
        err_red = 1 - model.score(test_data_red.iloc[:,0:ncov], test_data_red.iloc[:,ncov+1])
        err_blue = 1 - model.score(test_data_blue.iloc[:,0:ncov], test_data_blue.iloc[:,ncov+1])
        red_val_errors = red_val_errors.append(pd.Series(err_red))
        blue_val_errors = blue_val_errors.append(pd.Series(err_blue))
        train_err = 1 - model.score(train_data.iloc[:,0:ncov],train_data.iloc[:,ncov+1])
        test_err = 1 - model.score(test_data.iloc[:,0:ncov],test_data.iloc[:,ncov+1])
        test_err_dif = err_blue-err_red
        training_errors = training_errors.append(pd.Series(train_err))
        test_errors = test_errors.append(pd.Series(test_err))
        test_err_difs = test_err_difs.append(pd.Series(test_err_dif))
        times = times.append(pd.Series(t))
        t= t+1
    errors = pd.DataFrame(
        {'red': red_val_errors, 'blue': blue_val_errors, 'train': training_errors, 'test': test_errors,
         'resid': test_err_difs, 'time': times})

    param_labels = str(ncov) + '_cov'+ str(p_red)+'_probred' + str(total_training)+'_trainsize'+str(test_size)+'_testsize'
    errors.plot(x='time', y=['blue', 'red'])
    plt.savefig('./experiments/'+param_labels+'_errors.png')
    plt.close
    errors.plot(x='time', y=['train', 'test'])
    plt.savefig('./experiments/'+param_labels+'_train_test.png')
    plt.close
    errors.plot(x='time',y='resid')
    plt.savefig('./experiments/'+param_labels+'_test_dif.png')

# run_experiment_gnostic_logistic_(5,.8,1000,100,1000)
#
# train_data = gen_data_dm(p_red, total_training, ncov, mu_red, mu_blue, sigma_red, sigma_blue, red_beta, blue_beta)
# test_data = gen_data_dm(p_red, test_size, ncov, mu_red, mu_blue, sigma_red, sigma_blue, red_beta, blue_beta)
# test_data_red = test_data[test_data.iloc[:, ncov] == 1]
# test_data_blue = test_data[test_data.iloc[:, ncov] == 2]
#
# dt = dtc(max_depth=1,max_features=1).fit(train_data.iloc[:,0:ncov], train_data.iloc[:,ncov+1])
# sum(dt.predict(train_data.iloc[:,0:ncov])==train_data.iloc[:,ncov+1])
# sum(dt.predict(test_data.iloc[:,0:ncov])==test_data.iloc[:,ncov+1])
# sum(dt.predict(test_data_red.iloc[:,0:ncov])==test_data_red.iloc[:,ncov+1])
# sum(dt.predict(test_data_blue.iloc[:,0:ncov])==test_data_blue.iloc[:,ncov+1])

class gnostic_fair_learner:
    def __init__(self,train_data,ncov, val_data_red, val_data_blue):
        self.wl = dtc(max_depth=1, max_features =1).fit(train_data.iloc[:,0:(ncov+1)], train_data.iloc[:,ncov+1])
        self.red_err = 1.0-dt.score(val_data_red.iloc[:,0:ncov],val_data_red.iloc[:,ncov+1])
        self.blue_err = 1.0-dt.score(val_data_blue.iloc[:,0:ncov],val_data_blue.iloc[:,ncov+1])
        self.prob = (self.red_err-self.blue_err)/1.0/(0.5-self.blue_err)
        def predict(sample,ncov):
            features = np.array(sample.iloc[0:ncov+1]).reshape(1,-1)
            prediction = self.wl.predict(X=features)[0]
            if features[ncov]==1:
                if np.random.rand() < self.prob:
                    prediction = np.random.choice([0,1])
            return prediction
        self.predict = predict

class gnostic_fair_learner:
    def __init__(self,train_data,ncov, val_data_red, val_data_blue,weights=None,fair=True,val_data_generator = None):
        train_data_red = train_data[train_data.iloc[:,ncov]==1]
        train_data_blue = train_data[train_data.iloc[:,ncov]==2]

        self.fair = fair
        self.wl = dtc(max_depth=1, max_features =1).fit(train_data.iloc[:,0:(ncov+1)], train_data.iloc[:,ncov+1],sample_weight=weights)
        # if val_data_generator is None:
        #     self.red_err = 1.0- self.wl.score(train_data_red.iloc[:, 0:ncov + 1], train_data_red.iloc[:, ncov + 1])
        #     self.blue_err = 1.0-self.wl.score(train_data_blue.iloc[:,0:ncov+1],train_data_blue.iloc[:,ncov+1])
        # else:
            #val_data_red = val_data_generator.gen_data(100,1.0)
            #val_data_blue = val_data_generator.gen_data(100,0.0)
        self.red_err = 1.0 - self.wl.score(val_data_red.iloc[:, 0:ncov + 1], val_data_red.iloc[:, ncov + 1])
        self.blue_err = 1.0 - self.wl.score(val_data_blue.iloc[:, 0:ncov + 1], val_data_blue.iloc[:, ncov + 1])
        self.color_less_error = 0
        if self.red_err < self.blue_err:
            self.color_less_error = 1.0
            self.prob = (self.blue_err - self.red_err) / 1.0 / (0.5 - self.red_err)
        if self.red_err > self.blue_err:
            self.color_less_error = 2.0
            self.prob = (self.blue_err - self.red_err) / 1.0 / (0.5 - self.red_err)
        else:
            self.prob = 0
        def return_red_err():
            return self.red_err
        def return_blue_err():
            return self.blue_err
        def return_wl():
            return self.wl
        def predict(sample,ncov):
            features = np.array(sample.iloc[0:ncov+1]).reshape(1,-1)
            prediction = self.wl.predict(X=features)[0]
            if self.fair==True:
                if features[0][ncov]==self.color_less_error:
                    if np.random.rand() < self.prob:
                        prediction = np.random.choice([0,1])
            return prediction
        def batch_predict(dataset,ncov):
            predictions = self.wl.predict(X=dataset.iloc[:,0:ncov+1])
            if self.fair == True:
                for i in range(dataset.shape[0]):
                    if dataset.iloc[i,ncov]==self.color_less_error:
                        if np.random.rand() < self.prob:
                            predictions[i] = np.random.choice([0,1])
            return predictions
        self.predict = predict
        self.batch_predict = batch_predict
        self.return_red_err = return_red_err
        self.return_blue_err = return_blue_err
        self.return_wl = return_wl


# gfl = gnostic_fair_learner(train_data, 5, test_data_red, test_data_blue,weights=None)
# gfl.batch_predict(train_data.iloc[:,0:ncov+1],ncov)
class data_generator:
    def __init__(self,p_red, n,ncov, params):
        self.p_red = p_red
        self.n = n
        self.ncov = ncov
        self.params = params
        def gen_data(samplesize=0,prob_red=0):
            ss= max(self.n, n)
            p = max(self.p_red,prob_red)
            return gen_data_dm_params(p, ss, self.ncov, self.params)
        self.gen_data = gen_data
class empirical_data_generator:
    def __init__(self, empirical_data):
        self.empirical_data = empirical_data
        def gen_data():
            return self.empirical_data
        self.gen_data = gen_data





# data_gen = data_generator(.5,100,5,mu_red,mu_blue, sigma_red,sigma_blue, red_beta,blue_beta)
def boost_fair(train_data,max_learner,ncov, validation_data_gen,fairness=True):
    len_data = train_data.shape[0]
    log_weights =np.log(1.0/len_data* np.ones((len_data)))
    learners = []
    alphas = []
    hypothesis = []
    l=1
    test_data = validation_data_gen.gen_data()

    while l < max_learner+1:
        train_data.reindex()
        td_indx = train_data.index
        print sum(np.exp(log_weights))
        val_sample_weights = np.exp(log_weights)/(sum(np.exp(log_weights)))
        val_indices = np.random.choice(td_indx, 100, p=val_sample_weights)
        val_data = train_data.loc[val_indices]
        val_data_red = val_data[val_data.iloc[:,ncov]==1]
        val_data_blue = val_data[val_data.iloc[:,ncov]==2]
        # test_data = validation_data_gen.gen_data()
        new_learner = gnostic_fair_learner(train_data, ncov, val_data_red,val_data_blue,weights = np.exp(log_weights),fair=fairness)
        predictions = new_learner.batch_predict(train_data,ncov)
        errors = predictions != train_data.iloc[:,ncov+1]
        weighted_err = np.dot(np.exp(log_weights),errors)
        alpha = 0.5*np.log((1.0-weighted_err)/(1.0*weighted_err))
        alphas.append(alpha)
        lognormalizer =np.log(sum(np.exp(log_weights)))
        new_log_weights = log_weights + alpha*(2.0*np.array(errors)-1.0) - lognormalizer
        log_weights = np.array(new_log_weights)
        #sample_indices = np.random.choice(range(len_data), p=np.array(weights)[0])
        #data.loc[sample_indices]
        learners.append(new_learner)
        l = l+1

    def combined_hypothesis(sample):
        predictions = [learners[i].predict(sample,ncov) for i in range(len(learners))]
        predictions_rescaled = np.array(predictions)*2-1.0
        return np.sign(np.dot(alphas,predictions_rescaled))
    return combined_hypothesis


def boost_fair_all_errors(train_data,max_learner,ncov, validation_data_gen,fairness=True):
    len_data = train_data.shape[0]
    log_weights =np.log(1.0/len_data* np.ones((len_data)))
    learners = []
    alphas = []
    l=1
    test_data = validation_data_gen.gen_data()
    indv_errs_red = []
    indv_errs_blue = []
    indv_errs_test = []
    while l < max_learner+1:
        train_data.reindex()
        td_indx = train_data.index
        print sum(np.exp(log_weights))
        val_sample_weights = np.exp(log_weights)/(sum(np.exp(log_weights)))
        val_indices = np.random.choice(td_indx, 100, p=val_sample_weights)
        val_data = train_data.loc[val_indices]
        val_data_red = val_data[val_data.iloc[:,ncov]==1]
        val_data_blue = val_data[val_data.iloc[:,ncov]==2]
        new_learner = gnostic_fair_learner(train_data, ncov, val_data_red,val_data_blue,weights = np.exp(log_weights),fair=fairness)
        predictions = new_learner.batch_predict(train_data,ncov)
        errors = predictions != train_data.iloc[:,ncov+1]
        weighted_err = np.dot(np.exp(log_weights),errors)
        alpha = 0.5*np.log((1.0-weighted_err)/(1.0*weighted_err))
        alphas.append(alpha)
        lognormalizer =np.log(sum(np.exp(log_weights)))
        new_log_weights = log_weights + alpha*(2.0*np.array(errors)-1.0) - lognormalizer
        log_weights = np.array(new_log_weights)
        #sample_indices = np.random.choice(range(len_data), p=np.array(weights)[0])
        #data.loc[sample_indices]
        learners.append(new_learner)
        l = l+1
        val_pred_red = np.array(new_learner.batch_predict(val_data_red,ncov))*2.0-1.0
        val_pred_blue = np.array(new_learner.batch_predict(val_data_blue,ncov))*2.0-1.0
        test_pred = np.array(new_learner.batch_predict(test_data,ncov))*2.0-1.0
        error_rate_red = 1.0-sum(val_pred_red != val_data_red.iloc[:,ncov+1])/1.0/len(val_pred_red)
        error_rate_blue = 1.0-sum(val_pred_blue != val_data_blue.iloc[:,ncov+1])/1.0/len(val_pred_blue)
        error_rate_test = 1.0-sum(test_pred != test_data.iloc[:,ncov+1])/1.0/len(test_data)
        indv_errs_red.append(error_rate_red)
        indv_errs_blue.append(error_rate_blue)
        indv_errs_test.append(error_rate_test)
    errors = {'learners':range(1,len(learners)+1), 'err_rate_red':indv_errs_red,'err_rate_blue':indv_errs_blue,'err_rate_test':indv_errs_test}
    return errors


def train_learner(log_weights,  train_data, ncov, val_data_red, val_data_blue, fairness=True):
    new_learner = gnostic_fair_learner(train_data, ncov, val_data_red, val_data_blue, weights=np.exp(log_weights),
                                       fair=fairness)
    predictions = new_learner.batch_predict(train_data, ncov)
    errors = predictions != train_data.iloc[:, ncov + 1]
    weighted_err = np.dot(np.exp(log_weights), errors)
    alpha = 0.5*np.log((1.0-weighted_err)/(1.0*weighted_err))
    return new_learner, alpha

def get_val_split(log_weights,val_data_source,val_size,ncov):
    val_data_source.reindex()
    td_indx = val_data_source.index
    val_sample_weights = np.exp(log_weights) / (sum(np.exp(log_weights)))
    val_indices = np.random.choice(td_indx, val_size, p=val_sample_weights)
    val_data = val_data_source.loc[val_indices]
    val_data_red = val_data[val_data.iloc[:, ncov] == 1]
    val_data_blue = val_data[val_data.iloc[:, ncov] == 2]
    return val_data_red,val_data_blue

def get_new_log_weights(log_weights,alpha,errors):
    lognormalizer = np.log(sum(np.exp(log_weights)))
    new_log_weights = log_weights + alpha * (2.0 * np.array(errors) - 1.0) - lognormalizer
    log_weights = np.array(new_log_weights)
    return log_weights


def boost_iterative(train_data,ncov, validation_data_gen, learners =[],alphas = [],current_log_weights = [],fairness=True,val_data_size=100):
    len_data = train_data.shape[0]
    val_data_source = validation_data_gen.gen_data()
    if len(current_log_weights)==0:
        current_log_weights =np.log(1.0/len_data* np.ones((len_data)))
    red_val_data, blue_val_data = get_val_split(current_log_weights, val_data_source,val_data_size,ncov)
    new_learner, new_alpha = train_learner(current_log_weights, train_data,ncov, red_val_data,blue_val_data,fairness)
    learners.append(new_learner)
    alphas.append(new_alpha)
    predictions = new_learner.batch_predict(train_data, ncov)
    errors = predictions != train_data.iloc[:, ncov + 1]
    new_log_weights = get_new_log_weights(current_log_weights,new_alpha, errors)
    return learners, alphas, new_log_weights



def make_combined_hypothesis(learners, alphas, num_learners,ncov):
    def combined_hypothesis(sample):
        predictions = [learners[i].batch_predict(sample, ncov) for i in range(num_learners)]
        predictions_rescaled = np.array(predictions) * 2.0 - 1.0
        combined_preds = [np.sign(np.dot(alphas[0:num_learners],[predictions_rescaled[i][n] for i in range(num_learners)]))for n in range(len(sample))]
        #return np.sign(np.dot(alphas[0:num_learners],predictions_rescaled))
        return combined_preds
    return combined_hypothesis

def make_combined_classifier(combined_hypothesis):
    def combined_classifier(sample):
        return (np.sign(combined_hypothesis(sample))+1.0)/2.0
    return combined_classifier

def run_iterative_boosting_experiment(ncov, p_red, train_data,test_data,val_data_size, params, max_learners, fairness=True):
    train_data_len = len(train_data)
    validation_data_gen = data_generator(p_red, train_data_len, ncov, params)
    learners = []
    alphas = []
    current_log_weights = []
    frac_blue = []
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
    return error_metrics_train, error_metrics_test,frac_blue
def run_iterative_boosting_paired_experiment(ncov, p_red,train_data,test_data,val_data_size,params, max_learners_fair,max_learners_vanilla):
    exp_labels = str(ncov) + '_cov'+ str(p_red)+'_probred' + str(total_training)+'_trainsize'+str(test_size)+'_testsize'+'_maxlearners'
    res_fair = run_iterative_boosting_experiment(ncov, p_red,train_data,test_data, val_data_size, params, max_learners_fair,fairness=True)
    res_vanilla = run_iterative_boosting_experiment(ncov, p_red,train_data,test_data, val_data_size, params, max_learners_vanilla,fairness=False)
    res_dfs_fair = get_res_df(res_fair)
    res_dfs_vanilla = get_res_df(res_vanilla)
    frac_blue_fair = res_fair[2]
    frac_blue_vanilla = res_vanilla[2]
    plot_exp(res_dfs_fair, 'fair',exp_labels)
    plot_exp(res_dfs_vanilla, 'vanilla', exp_labels)
    plot_err_ratios_together('ratio', res_dfs_fair['train_err'],res_dfs_vanilla['train_err'],exp_labels, 'train')
    plot_err_ratios_together('ratio',res_dfs_fair['test_err'],res_dfs_vanilla['test_err'],exp_labels,'test')
    plot_err_ratios_together('log_exploss_dif', res_dfs_fair['train_exp'],res_dfs_vanilla['train_exp'],exp_labels, 'train')
    plot_err_ratios_together('log_exploss_dif',res_dfs_fair['test_exp'],res_dfs_vanilla['test_exp'],exp_labels,'test')
    return(res_dfs_fair, res_dfs_vanilla)
def calc_exponential_loss(data_w_pred,ncov):
    data_red = data_w_pred[data_w_pred.iloc[:,ncov]==1]
    data_blue = data_w_pred[data_w_pred.iloc[:,ncov]==2]
    total_exp_loss =np.log(np.sum(np.exp(np.multiply(data_w_pred.iloc[:,ncov+1],data_w_pred['margin']))))- np.log(len(data_w_pred))
    red_exp_loss =np.log(np.sum(np.exp(np.multiply(data_red.iloc[:,ncov+1],data_red['margin']))))- np.log(len(data_red))
    blue_exp_loss =np.log(np.sum(np.exp(np.multiply(data_blue.iloc[:,ncov+1],data_blue['margin']))))- np.log(len(data_blue))
    return {'log_total_exp': total_exp_loss, 'log_red_exp':red_exp_loss, 'log_blue_exp': blue_exp_loss}


def calc_accuracy(data_w_pred,ncov):
    data_red = data_w_pred[data_w_pred.iloc[:,ncov]==1]
    data_blue = data_w_pred[data_w_pred.iloc[:,ncov]==2]
    total_accuracy = sum(data_w_pred.iloc[:,ncov+1] == data_w_pred['pred'])/ 1.0/len(data_w_pred)
    red_accuracy = sum(data_red.iloc[:,ncov+1]==data_red['pred'])/1.0/len(data_red)
    blue_accuracy = sum(data_blue.iloc[:,ncov+1]==data_blue['pred'])/1.0/len(data_blue)
    return {'total':total_accuracy,'red': red_accuracy,'blue':blue_accuracy}
def get_metrics(combined_classifier,combined_hypothesis, train_data, test_data,ncov):
    train_predictions = combined_classifier(train_data)
    test_predictions = combined_classifier(test_data)
    train_margin = combined_hypothesis(train_data)
    test_margin = combined_hypothesis(test_data)
    copy_train_data = train_data.copy(deep=True)
    copy_test_data = test_data.copy(deep=True)

    copy_train_data['pred'] = (np.array(train_predictions) + 1.0) * 0.5
    copy_test_data['pred'] = (np.array(test_predictions) + 1.0) * 0.5
    copy_train_data['margin'] = train_margin
    copy_test_data['margin'] = test_margin
    accuracy_train = calc_accuracy(copy_train_data,ncov)
    accuracy_test = calc_accuracy(copy_test_data,ncov)
    exp_loss_train = calc_exponential_loss(copy_train_data,ncov)
    exp_loss_test = calc_exponential_loss(copy_test_data,ncov)
    return accuracy_train, accuracy_test,exp_loss_train, exp_loss_test
def get_res_df(res):
    train_accs = pd.DataFrame.from_dict(res[0]['accs'])
    train_errs = 1.0-train_accs
    train_err_ratio = train_errs['blue']/1.0/train_errs['red']
    train_errs['ratio'] = train_err_ratio

    test_accs = pd.DataFrame.from_dict(res[1]['accs'])
    test_errs = 1.0-test_accs
    test_err_ratio = test_errs['blue']/1.0/test_errs['red']
    test_errs['ratio'] = test_err_ratio

    train_exps = pd.DataFrame.from_dict(res[0]['exp'])
    train_log_difs = train_exps['log_red_exp']-train_exps['log_blue_exp']

    test_exps = pd.DataFrame.from_dict(res[1]['exp'])
    test_log_difs = test_exps['log_red_exp']-test_exps['log_blue_exp']



    learners_df = pd.DataFrame.from_dict(res[0]['learners'])


    train_acc_df = pd.concat([learners_df,train_accs], axis=1)
    train_acc_df.columns = ['learners', 'blue_acc', 'red_acc', 'train_acc']
    train_err_df = pd.concat([learners_df, train_errs],axis=1)
    train_err_df.columns = ['learners','blue_err','red_err','train_err','ratio']
    test_acc_df = pd.concat([learners_df, test_accs], axis=1)
    test_acc_df.columns = ['learners', 'blue_acc', 'red_acc', 'test_acc']
    test_err_df = pd.concat([learners_df, test_errs],axis=1)
    test_err_df.columns = ['learners','blue_err','red_err','test_err','ratio']
    train_exp_df = pd.concat([learners_df, train_log_difs],axis=1)
    train_exp_df.columns = ['learners', 'log_exploss_dif']
    test_exp_df = pd.concat([learners_df, test_log_difs],axis=1)
    test_exp_df.columns = ['learners', 'log_exploss_dif']

    return {'train_acc': train_acc_df,'train_err': train_err_df, 'test_acc': test_acc_df, 'test_err': test_err_df,'train_exp': train_exp_df,'test_exp':test_exp_df}

def plot_exp(dfs, fair_str, param_labels):
    train_acc_df = dfs['train_acc']
    train_err_df = dfs['train_err']
    test_acc_df = dfs['test_acc']
    test_err_df = dfs['test_err']
    train_exp_df = dfs['train_exp']
    test_exp_df = dfs['test_exp']

    train_acc_df.plot(x='learners', y=['blue_acc', 'red_acc', 'train_acc'])
    plt.savefig('./experiments/'+fair_str+'_boosting' + param_labels + '_train_acc.png')
    plt.close()

    train_err_df.plot(x='learners', y=['blue_err', 'red_err', 'train_err'])
    plt.savefig('./experiments/' + fair_str+'_boosting' + param_labels + '_train_err.png')
    plt.close()

    train_err_df.plot(x='learners', y=['ratio'])
    plt.savefig('./experiments/' + fair_str+'_boosting' + param_labels + '_train_err_ratios.png')
    plt.close()

    train_exp_df.plot(x='learners', y=['log_exploss_dif'])
    plt.savefig('./experiments/'+fair_str+'_boosting' + param_labels + '_train_exp_loss.png')
    plt.close()


    test_acc_df.plot(x='learners', y=['blue_acc', 'red_acc', 'test_acc'])
    plt.savefig('./experiments/'+fair_str+'_boosting' + param_labels + '_test_acc.png')
    plt.close()

    test_err_df.plot(x='learners', y=['blue_err', 'red_err', 'test_err'])
    plt.savefig('./experiments/' + fair_str+'_boosting' + param_labels + '_test_err.png')
    plt.close()

    test_err_df.plot(x='learners', y=['ratio'])
    plt.savefig('./experiments/' + fair_str+'_boosting' + param_labels + '_test_err_ratios.png')
    plt.close()

    test_exp_df.plot(x='learners', y=['log_exploss_dif'])
    plt.savefig('./experiments/'+fair_str+'_boosting' + param_labels + '_test_exp_loss.png')
    plt.close()

def plot_err_ratios_together(var_string,fair_err_df, vanilla_err_df,param_labels, test_str):
    fair_res = fair_err_df[var_string]
    vanilla_res = vanilla_err_df[var_string]
    n_l_fair = len(fair_res)
    n_l_vanilla = len(vanilla_res)
    fair_varname = 'fair_'+var_string
    vanilla_varname = 'vanilla_'+var_string
    data_dict = {'learners': range(1,max(n_l_fair,n_l_vanilla)+1), fair_varname :fair_res , vanilla_varname :vanilla_res}
    graph_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
    graph_data.fillna(inplace=True, method='ffill')
    graph_data.plot(x='learners',y=[fair_varname,vanilla_varname])
    plt.savefig('./experiments/' + '_boosting'+ param_labels + '_' + test_str +'err'+var_string +'.png')
    plt.close()





# learners = boost_fair(train_data, 2, 5, data_gen)

def run_experiment_boosting(ncov,p_red,total_training,test_size,exp_len,fairness=True,train_data = None,params=None, plot=True):

    param_labels = str(ncov) + '_cov'+ str(p_red)+'_probred' + str(total_training)+'_trainsize'+str(test_size)+'_testsize'+str(exp_len)+'_maxlearners'

    if train_data is None:
        mu_red = np.random.rand(ncov)
        mu_blue = np.random.rand(ncov)
        sigma_red = .3
        sigma_blue = .3
        red_beta = np.random.randn(ncov)
        blue_beta = np.random.randn(ncov)
        params = {'mu_red': mu_red, 'mu_blue':mu_blue, 'sigma_red': sigma_red, 'sigma_blue': sigma_blue, 'beta_red':red_beta, 'beta_blue':blue_beta}
        train_data = gen_data_dm_params(p_red, total_training, ncov, params)
    else:
        mu_red = params['mu_red']
        mu_blue = params['mu_blue']
        sigma_red = params['sigma_red']
        sigma_blue = params['sigma_blue']
        red_beta = params['beta_red']
        blue_beta = params['beta_blue']

    validation_data_gen = data_generator(p_red, test_size, ncov, mu_red, mu_blue, sigma_red, sigma_blue, red_beta, blue_beta)
    max_no_learners = exp_len
    red_accs = []
    red_test_accs = []
    blue_test_accs = []
    blue_accs = []
    learners = []
    train_accuracy = []
    test_accuracy = []
    num_learners = 1
    red_indv_test_accs = []
    blue_indv_test_accs = []
    test_data = gen_data_dm_params(p_red, test_size, ncov, params)
    train_data_red = train_data[train_data.iloc[:, ncov] == 1]
    train_data_blue = train_data[train_data.iloc[:, ncov] == 2]
    test_data_red = test_data[test_data.iloc[:, ncov] == 1]
    test_data_blue = test_data[test_data.iloc[:, ncov] == 2]

    while(num_learners<max_no_learners):
        print num_learners
        print(datetime.datetime.now())
        #test_data = gen_data_dm_params(p_red,test_size, ncov, params)
        #test_data = gen_data_dm(p_red, test_size, ncov, mu_red, mu_blue, sigma_red, sigma_blue, red_beta, blue_beta)
        hypothesis = boost_fair(train_data, num_learners, ncov, validation_data_gen,fairness=fairness)
        h_predictions = [hypothesis(train_data.iloc[i,0:ncov+1]) for i in range(len(train_data))]
        h_test_red_predictions = [hypothesis(test_data_red.iloc[i,0:ncov+1]) for i in range(len(test_data_red))]
        h_test_blue_predictions = [hypothesis(test_data_blue.iloc[i,0:ncov+1]) for i in range(len(test_data_blue))]
        h_test_predictions = [hypothesis(test_data.iloc[i,0:ncov+1]) for i in range(len(test_data))]
        h_train_red_predictions = [hypothesis(train_data_red.iloc[i,0:ncov+1]) for i in range(len(train_data_red))]
        h_train_blue_predictions = [hypothesis(train_data_blue.iloc[i,0:ncov+1]) for i in range(len(train_data_blue))]
        test_data['pred'] = (np.array(h_test_predictions)+1)*0.5
        train_data['pred'] = (np.array(h_predictions)+1)*0.5
        test_data_red['pred'] =(np.array(h_test_red_predictions) +1)*0.5
        test_data_blue['pred'] = (np.array(h_test_blue_predictions) +1)*0.5
        train_data_red['pred'] = (np.array(h_train_red_predictions)+1)*0.5
        train_data_blue['pred'] = (np.array(h_train_blue_predictions)+1)*0.5
        err_train = sum(train_data['pred'] != train_data.iloc[:,ncov+1])/1.0/len(train_data)
        err_test = sum(test_data['pred'] != test_data.iloc[:,ncov+1])/1.0/len(test_data)
        err_red_test = sum(test_data_red['pred'] != test_data_red.iloc[:,ncov+1])/1.0/len(test_data_red)
        err_blue_test = sum(test_data_blue['pred'] != test_data_blue.iloc[:,ncov+1])/1.0/len(test_data_blue)
        err_red = sum(train_data_red['pred'] != train_data_red.iloc[:,ncov+1])/1.0/len(train_data_red)
        err_blue = sum(train_data_blue['pred'] != train_data_blue.iloc[:,ncov+1])/1.0/len(train_data_blue)
        red_acc = 1.0-err_red
        blue_acc = 1.0-err_blue
        train_acc = 1.0-err_train
        red_test_acc = 1.0-err_red_test
        blue_test_acc = 1.0-err_blue_test
        test_acc = 1.0-err_test
        red_accs.append(red_acc)
        red_test_accs.append(red_test_acc)
        blue_test_accs.append(blue_test_acc)
        test_accuracy.append(test_acc)
        blue_accs.append(blue_acc)
        train_accuracy.append(train_acc)
        learners.append(num_learners)
        num_learners = num_learners+1
    errors = pd.DataFrame(
        {'red_acc': red_accs, 'blue_acc': blue_accs, 'train_acc': train_accuracy, 'red_test': red_test_accs, 'blue_test':blue_test_accs, 'test_acc': test_accuracy, 'learners': learners})

    if plot==True:
        try:
            ax = errors.plot(x='learners', y=['blue_acc', 'red_acc', 'train_acc'])
            plt.savefig('./experiments/fairness_set'+ str(test_data_freshness)+str(fairness)+'_boosting'+param_labels+'_train_errors.png')
            plt.close
            ax = errors.plot(x='learners', y=['blue_test', 'red_test', 'test_acc'])
            plt.savefig('./experiments/fairness_set'+ str(test_data_freshness)+str(fairness)+'_boosting'+param_labels+'_test_errors.png')
            plt.close

        except:
            print 'exception! in plotting'
        finally:
            return errors
    return errors





def run_multiple_boosting_experiments(max_num_experiments, num_exp_to_avg, ncov,p_red,total_training,test_size):
    results_fair = {'learners': [], 'avg_red_acc':[], 'avg_blue_acc': [], 'avg_train':[], 'avg_red_test_acc':[], 'avg_blue_test_acc':[], 'avg_test':[], 'train_ratios':[], 'test_ratios':[]}
    results_vanilla = {'learners': [], 'avg_red_acc':[], 'avg_blue_acc': [], 'avg_train':[], 'avg_red_test_acc':[], 'avg_blue_test_acc':[], 'avg_test':[], 'train_ratios':[], 'test_ratios':[]}
    for i in range(1,max_num_experiments+1):
        print str(i) + ' learners'
        print(datetime.datetime.now())
        errs_fair = []
        errs_vanilla = []
        for j in range(1,num_exp_to_avg+1):
            try:
                mu_red_fixed = np.random.rand(ncov)
                mu_blue_fixed = np.random.rand(ncov)
                sigma_red_fixed = .3
                sigma_blue_fixed = .3
                red_beta_fixed = np.random.randn(ncov)
                blue_beta_fixed = np.random.randn(ncov)
                rand_params = {'mu_red': mu_red_fixed, 'mu_blue': mu_blue_fixed, 'sigma_red': sigma_red_fixed,
                                'sigma_blue': sigma_blue_fixed, 'beta_red': red_beta_fixed, 'beta_blue': blue_beta_fixed}

                random_train_data = gen_data_dm_params(p_red, total_training,ncov, rand_params)

                fair_exp = run_experiment_boosting(ncov,p_red,total_training,test_size,i+1,fairness = True,train_data=random_train_data,params = rand_params,plot=False)
                vanilla_exp = run_experiment_boosting(ncov,p_red,total_training,test_size,i+1,fairness = False,train_data=random_train_data,params = rand_params,plot=False)
                errs_fair.append(fair_exp)
                errs_vanilla.append(vanilla_exp)
            except:
                print 'exception! in boosting experiment j loop'
        fair_avg_red_acc = np.mean([errs_fair[k]['red_acc'] for k in range(len(errs_fair))])
        fair_avg_blue_acc = np.mean([errs_fair[k]['blue_acc'] for k in range(len(errs_fair))])
        fair_avg_train_acc = np.mean([errs_fair[k]['train_acc'] for k in range(len(errs_fair))])
        fair_avg_red_test_acc = np.mean([errs_fair[k]['red_test'] for k in range(len(errs_fair))])
        fair_avg_blue_test_acc = np.mean([errs_fair[k]['blue_test'] for k in range(len(errs_fair))])
        fair_avg_test_acc = np.mean([errs_fair[k]['test_acc'] for k in range(len(errs_fair))])
        fair_train_ratios = fair_avg_blue_acc / fair_avg_red_acc
        fair_test_ratios = fair_avg_blue_test_acc / fair_avg_red_test_acc
        vanilla_avg_red_acc = np.mean([errs_vanilla[k]['red_acc'] for k in range(len(errs_vanilla))])
        vanilla_avg_blue_acc = np.mean([errs_vanilla[k]['blue_acc'] for k in range(len(errs_vanilla))])
        vanilla_avg_train_acc = np.mean([errs_vanilla[k]['train_acc'] for k in range(len(errs_vanilla))])
        vanilla_avg_red_test_acc = np.mean([errs_vanilla[k]['red_test'] for k in range(len(errs_vanilla))])
        vanilla_avg_blue_test_acc = np.mean([errs_vanilla[k]['blue_test'] for k in range(len(errs_vanilla))])
        vanilla_avg_test_acc = np.mean([errs_vanilla[k]['test_acc'] for k in range(len(errs_vanilla))])
        vanilla_train_ratios = vanilla_avg_blue_acc / vanilla_avg_red_acc
        vanilla_test_ratios = vanilla_avg_blue_test_acc / vanilla_avg_red_test_acc
        results_fair['learners'].append(i)
        results_fair['avg_red_acc'].append(fair_avg_red_acc)
        results_fair['avg_blue_acc'].append(fair_avg_blue_acc)
        results_fair['avg_train'].append(fair_avg_train_acc)
        results_fair['avg_test'].append(fair_avg_test_acc)
        results_fair['avg_red_test_acc'].append(fair_avg_red_test_acc)
        results_fair['avg_blue_test_acc'].append(fair_avg_blue_test_acc)
        results_fair['train_ratios'].append(fair_train_ratios)
        results_fair['test_ratios'].append(fair_test_ratios)
        results_vanilla['learners'].append(i)
        results_vanilla['avg_red_acc'].append(vanilla_avg_red_acc)
        results_vanilla['avg_blue_acc'].append(vanilla_avg_blue_acc)
        results_vanilla['avg_train'].append(vanilla_avg_train_acc)
        results_vanilla['avg_test'].append(vanilla_avg_test_acc)
        results_vanilla['avg_red_test_acc'].append(vanilla_avg_red_test_acc)
        results_vanilla['avg_blue_test_acc'].append(vanilla_avg_blue_test_acc)
        results_vanilla['train_ratios'].append(vanilla_train_ratios)
        results_vanilla['test_ratios'].append(vanilla_test_ratios)

    fair_results = pd.DataFrame(results_fair)
    vanilla_results = pd.DataFrame(results_vanilla)
    try:
        fair_results.plot(x='learners', y=['avg_blue_acc', 'avg_red_acc', 'avg_train'])
        plt.savefig('./experiments/fair_boosting_set'+ str(test_data_freshness) +str(datetime.datetime.now())+'avg_train_errors.png')
        plt.close()
        fair_results.plot(x='learners', y=['avg_blue_test_acc', 'avg_red_test_acc', 'avg_test'])
        plt.savefig('./experiments/fair_boosting_set'+ str(test_data_freshness)+str(datetime.datetime.now())+'avg_test_errors.png')
        plt.close()
        fair_results.plot(x='learners', y=['train_ratios','test_ratios'])
        plt.savefig('./experiments/fair_boosting_set'+ str(test_data_freshness)+str(datetime.datetime.now())+'ratios.png')
        plt.close()
        vanilla_results.plot(x='learners', y=['avg_blue_acc', 'avg_red_acc', 'avg_train'])
        plt.savefig('./experiments/vanilla_boosting_set'+ str(test_data_freshness)+str(datetime.datetime.now())+'avg_train_errors.png')
        plt.close()
        vanilla_results.plot(x='learners', y=['avg_blue_test_acc', 'avg_red_test_acc', 'avg_test'])
        plt.savefig('./experiments/vanilla_boosting_set'+ str(test_data_freshness)+str(datetime.datetime.now())+'avg_test_errors.png')
        plt.close()
        vanilla_results.plot(x='learners', y=['train_ratios','test_ratios'])
        plt.savefig('./experiments/vanilla_boosting_set'+ str(test_data_freshness)+str(datetime.datetime.now())+'ratios.png')
        plt.close()


    except:
        print 'exception!in plotting'
    finally:
        fair_results.to_csv('./experiments/fair_boosting_results' + str(datetime.datetime.now())+'.csv')
        vanilla_results.to_csv('./experiments/vanilla_boosting_results' + str(datetime.datetime.now())+'.csv')
        return fair_results,vanilla_results


#)

#jawns = run_experiment_gnostic_logistic_(5,.8,1000,100,1000)

if __name__ == '__main__':
    max_num_experiments = 20
    num_exp_to_avg = 5
    ncov = 5
    p_red = .8
    total_training=1000
    test_size = 1000
    mu_red_fixed = np.random.rand(ncov)
    mu_blue_fixed = np.random.rand(ncov)
    sigma_red_fixed = .3
    sigma_blue_fixed = .3
    red_beta_fixed = np.random.randn(ncov)
    blue_beta_fixed = np.random.randn(ncov)
    params_fixed  = {'mu_red': mu_red_fixed, 'mu_blue':mu_blue_fixed, 'sigma_red': sigma_red_fixed, 'sigma_blue': sigma_blue_fixed, 'beta_red':red_beta_fixed, 'beta_blue': blue_beta_fixed}

    # fixed_train_sample = gen_data_dm_params(.8, 1000, 5, params_fixed)
    # print 'starting exp 1'
    # wafs = run_experiment_boosting(5,.8,1000,1000,50,True,fixed_train_sample,params_fixed)
    # print 'starting exp 2'
    # wafs2 = run_experiment_boosting(5,.8,1000,1000,50,False,fixed_train_sample,params_fixed)
    # print 'starting exp 3'
    # wafs3 = run_multiple_boosting_experiments(max_num_experiments, num_exp_to_avg, ncov,p_red,total_training,test_size)
    #wafs3[1].to_csv('vanilla_exp_5_exp_1ktrain_1ktest.csv')
    #wafs3[0].to_csv('fair_exp_5_exp_1ktrain_1ktest.csv')

    max_num_experiments = 20
    num_exp_to_avg = 3
    ncov = 3
    p_red = .8
    total_training=100
    test_size = 1000
    max_learners_fair= 10
    max_learners_vanilla = 10
    val_data_size=1000
    mu_red_fixed = np.random.rand(ncov)
    mu_blue_fixed = np.random.rand(ncov)
    sigma_red_fixed = .3
    sigma_blue_fixed = .3
    red_beta_fixed = np.random.randn(ncov)
    blue_beta_fixed = red_beta_fixed
    params_fixed  = {'mu_red': mu_red_fixed, 'mu_blue':mu_blue_fixed, 'sigma_red': sigma_red_fixed, 'sigma_blue': sigma_blue_fixed, 'beta_red':red_beta_fixed, 'beta_blue': blue_beta_fixed}
    fixed_train_sample = gen_data_dm_params(p_red, total_training, ncov, params_fixed)
    fixed_test_sample = gen_data_dm_params(p_red, test_size, ncov, params_fixed)

    print 'starting exp 1'
    wafs = run_experiment_boosting(ncov,p_red,total_training,test_size,50,fairness=True, train_data=fixed_train_sample, params=params_fixed)
    print 'starting exp 2'
    wafs2 = run_experiment_boosting(ncov,p_red,total_training,test_size,50,fairness=False, train_data=fixed_train_sample,params=params_fixed)

    print 'starting exp 3'
    wafs3 = run_multiple_boosting_experiments(max_num_experiments, num_exp_to_avg, ncov,p_red,total_training,test_size)

    validation_data_gen = data_generator(p_red, test_size, ncov, mu_red_fixed, mu_blue_fixed, sigma_red_fixed, sigma_blue_fixed, red_beta_fixed, blue_beta_fixed)

    errors_vanilla = boost_fair_all_errors(fixed_train_sample, 30, ncov, validation_data_gen, fairness=False)
    pd.DataFrame(errors_vanilla).plot(x='learners',y=['err_rate_blue','err_rate_red','err_rate_test'])
    plt.savefig('./experiments/vanilla_boosting_error' + str(test_data_freshness) + str(datetime.datetime.now()) + '.png')

    errors_fair = boost_fair_all_errors(fixed_train_sample, 30, ncov, validation_data_gen, fairness=True)
    pd.DataFrame(errors_fair).plot(x='learners',y=['err_rate_blue','err_rate_red','err_rate_test'])
    plt.savefig('./experiments/fairness_boosting_error' + str(test_data_freshness) + str(datetime.datetime.now()) +'.png')


    wafs3[0]['test_ratios']

    res = run_iterative_boosting_experiment(ncov, p_red, fixed_train_sample, fixed_test_sample, val_data_size, params_fixed, max_learners,fairness=True)

    # params = params_fixed
    # train_data = fixed_train_sample
    # test_data = fixed_test_sample

    max_num_experiments = 20
    num_exp_to_avg = 3
    ncov = 3
    p_red = .8
    total_training = 1000
    test_size = 1000
    max_learners_fair = 100
    max_learners_vanilla = 50
    val_data_size = 1000
    mu_red_fixed = np.random.rand(ncov)
    mu_blue_fixed = np.random.rand(ncov)
    sigma_red_fixed = .3
    sigma_blue_fixed = .3
    red_beta_fixed = np.random.randn(ncov)
    blue_beta_fixed = red_beta_fixed
    params_fixed = {'mu_red': mu_red_fixed, 'mu_blue': mu_blue_fixed, 'sigma_red': sigma_red_fixed,
                    'sigma_blue': sigma_blue_fixed, 'beta_red': red_beta_fixed, 'beta_blue': blue_beta_fixed}
    fixed_train_sample = gen_data_dm_params(p_red, total_training, ncov, params_fixed)
    fixed_test_sample = gen_data_dm_params(p_red, test_size, ncov, params_fixed)
    res = run_iterative_boosting_paired_experiment(ncov, p_red, fixed_train_sample, fixed_test_sample, val_data_size, params_fixed,
                                             max_learners_fair, max_learners_vanilla)
    results = []
    for j in range(5):
        max_num_experiments = 20
        num_exp_to_avg = 3
        ncov = 3
        p_red = .8
        total_training = 1000
        test_size = 1000
        max_learners_fair = 30
        max_learners_vanilla = 30
        val_data_size = 1000
        mu_red_fixed = np.random.rand(ncov)
        mu_blue_fixed = np.random.rand(ncov)
        sigma_red_fixed = .3
        sigma_blue_fixed = .3
        red_beta_fixed = np.random.randn(ncov)
        blue_beta_fixed = red_beta_fixed
        params_fixed = {'mu_red': mu_red_fixed, 'mu_blue': mu_blue_fixed, 'sigma_red': sigma_red_fixed,
                        'sigma_blue': sigma_blue_fixed, 'beta_red': red_beta_fixed, 'beta_blue': blue_beta_fixed}
        fixed_train_sample = gen_data_dm_params(p_red, total_training, ncov, params_fixed)
        fixed_test_sample = gen_data_dm_params(p_red, test_size, ncov, params_fixed)

        res_jawn = run_iterative_boosting_paired_experiment(ncov, p_red, fixed_train_sample, fixed_test_sample,
                                                       val_data_size, params_fixed,
                                                       max_learners_fair, max_learners_vanilla)
        results.append(res_jawn)

