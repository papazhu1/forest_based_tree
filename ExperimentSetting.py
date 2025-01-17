import numpy as np
import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from ConjunctionSet import ConjunctionSet
from Branch import Branch
from DataPreperation import *
import os
from ReadDatasetFunctions import *
from NewModelBuilder import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
import pickle
from rich.progress import Progress
import time

class ExperimentSetting():
    def __init__(self,number_of_branches_threshold,df_names,number_of_estimators,fixed_params,
                 num_of_iterations=1):
        self.num_of_iterations=num_of_iterations # 30
        self.number_of_branches_threshold = number_of_branches_threshold # [3000]
        self.df_names = df_names # ['iris']
        self.fixed_params = fixed_params # {}
        self.number_of_estimators = number_of_estimators # 100
    def run(self):
        self.experiments = []
        for threshold in self.number_of_branches_threshold: # 现在就只有一个[3000]
            for df_name in self.df_names: # 现在就只有一个['iris']
                df, x_columns, y_column, feature_types = get_dataset_by_string(df_name)
                print("df_name: ", df_name)
                print("len_x_columns: ", len(x_columns), " len_y_column: ", len(y_column), " feature_types: ", feature_types)
                d = {}
                d['max_number_of_branches'] = threshold
                d['df_name'] = df_name
                d['number_of_estimators'] = self.number_of_estimators
                print(d)

                # 传给run_experiment的参数有
                # 最多分支数，数据集，X的列名，y的列名，X的类型，特征类型，数据集名称，每个森林的决策树数量
                self.run_experiment(threshold, df, x_columns, y_column, feature_types, d)
    def run_experiment(self,branch_probability_threshold,df,x_columns,y_column,feature_types,hyper_parameters_dict):
        progress = Progress()
        task1 = progress.add_task(hyper_parameters_dict['df_name'] + '_iterate', total=self.num_of_iterations)
        progress.start()
        for i in range(self.num_of_iterations):
            tic = time.time()
            print(i)
            np.random.seed(i)
            num_of_estimators=hyper_parameters_dict['number_of_estimators']
            result_dict=dict(hyper_parameters_dict)
            result_dict['iteration']=i
            output_path = 'pickles_200trees/' + str(hyper_parameters_dict['df_name']) + '_' + str(result_dict['iteration'])
            if os.path.isfile(output_path):
                continue

            trainAndValidation_x, trainAndValidation_y, test_x, test_y = divide_to_train_test(df, x_columns, y_column)
            train_x = trainAndValidation_x[:int(len(trainAndValidation_x) * 0.8)]
            train_y = trainAndValidation_y[:int(len(trainAndValidation_x) * 0.8)]
            validation_x = trainAndValidation_x[int(len(trainAndValidation_x) * 0.8):]
            validation_y = trainAndValidation_y[int(len(trainAndValidation_x) * 0.8):]



            #Train random forest
            start_temp=datetime.datetime.now()
            rf = RandomForestClassifier(n_estimators=num_of_estimators,max_depth=5,min_samples_leaf=max(1,int(0.02*len(train_x))), **self.fixed_params)
            #rf = ExtraTreesClassifier(n_estimators=num_of_estimators, max_depth=3,min_samples_leaf=max(1, int(0.02 * len(train_x))), **self.fixed_params)
            rf.fit(trainAndValidation_x, trainAndValidation_y)
            result_dict['random forest training time']=(datetime.datetime.now()-start_temp).total_seconds()
            self.classes_=rf.classes_

            #Create the conjunction set
            start_temp = datetime.datetime.now()
            cs = ConjunctionSet(progress, x_columns, trainAndValidation_x,trainAndValidation_x,trainAndValidation_y, rf, feature_types,
                                hyper_parameters_dict['max_number_of_branches'])
            result_dict['conjunction set training time'] = (datetime.datetime.now() - start_temp).total_seconds()
            result_dict['number of branches per iteration'] = cs.number_of_branches_per_iteration
            result_dict['number_of_branches'] = len(cs.conjunctionSet)

            #Train the new model
            start_temp = datetime.datetime.now()
            branches_df = cs.get_conjunction_set_df().round(decimals=6)

            # branches_df 的 columns_name
            # branches_df.columns: Index(['0_upper', '0_lower', '1_upper', '1_lower', '2_upper', '2_lower',
            # '3_upper', '3_lower', 'number_of_samples', 'branch_probability',
            # 'probas'], dtype='object')
            print("branches_df:")
            print(branches_df)
            branches_df = branches_df.round(6)
            result_dict['number_of_features_for_new_model'] = len(branches_df.columns)
            
            # 原本是2，我想着iris数据集有3个类别，所以改成3
            # 对branches_df加了三列，分别是[setosa][versicolor][virginica]每列上的数值代表了与该分支匹配的示例对应三个类别的概率
            for j in range(3): 
                branches_df[rf.classes_[j]] = [probas[j] for probas in branches_df['probas']]
                # print("branches_df[rf.classes_[i]]: ")
                # print(branches_df[rf.classes_[i]])
                # print()
            print("branches_df_new:")
            print(branches_df)
            print("branches_df['probas'].mean()")
            print(branches_df['probas'].mean())
            df_dict = {}
            for col in branches_df.columns:
                # 对DataFrame取一整列是Series类型，是有索引+数值的，而加.values()后就变成了array类型，纯数值型
                df_dict[col] = branches_df[col].values # 将branches_df的每一列转换成array类型
            new_model = Node([True]*len(branches_df)) # 这是决策树的根节点
            print('start_split')
            new_model.split(df_dict) # 这是一个递归生成树的过程
            print('finish_split')
            result_dict['new model training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            #Train a decision tree
            start_temp = datetime.datetime.now()
            decision_tree_model=self.fit_decision_tree_model(trainAndValidation_x, trainAndValidation_y)
            result_dict['decision tree training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            #Train CMM tree
            '''
            start_temp = datetime.datetime.now()
            cmm_data = pd.DataFrame(trainAndValidation_x,columns=x_columns)
            cmm_data[y_column] = trainAndValidation_y
            cmm_dt = self.fit_cmm_tree(cmm_data,x_columns,y_column,rf)
            result_dict['cmm tree training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            '''
            
            # 对随机森林进行评估
            rusult_rf = self.ensemble_measures(test_x,test_y,rf)
            
            # 这个是合并branch变成决策树的新模型
            result_new_model = self.new_model_measures(test_x,test_y,new_model,branches_df)

            # 如果单棵决策树都比new_model好，那就直接使用决策树就好了
            result_dt = self.decision_tree_measures(test_x,test_y,decision_tree_model)

            # result_dict.update(self.cmm_tree_measures(test_x,test_y,cmm_dt))

            result_list = [rusult_rf,result_new_model,result_dt]
            result_df = pd.DataFrame(result_list)
            index_1 = 'iter_{}_rf'.format(i)
            index_2 = 'iter_{}_new_model'.format(i)
            index_3 = 'iter_{}_dt'.format(i)
            result_df.index = [index_1, index_2, index_3]

            # 将概率向量列表去除掉，因为没办法用csv文件来分析
            result_df = result_df.drop('probas',axis=1)
            print("result_df:")
            print(result_df)

            with open(output_path,'wb') as fp:
                pickle.dump(result_dict, fp)
            self.experiments.append(result_dict)
            progress.update(task1, advance=1)

            toc = time.time()
            elapsed_time = toc - tic
            print('Cur Iteration Time Elapsed: ', elapsed_time)

            with open('result.csv', 'a') as f:
                if i == 0:
                    result_df.to_csv(f, header=True)
                else:                    
                    result_df.to_csv(f, header=False)

        
        progress.stop()
    def decision_tree_measures(self,X,Y,dt_model):
        result_dict={}
        probas=[]
        depths=[]
        for inst in X:
            # 记录树的深度和预测结果
            pred,dept=self.tree_depth_and_prediction(inst,dt_model.tree_)
            probas.append(pred)
            depths.append(dept)
        predictions=dt_model.predict(X)
        result_dict['average_depth'] = np.mean(depths)
        result_dict['min_depth'] = np.min(depths)
        result_dict['max_depth'] = np.max(depths)

        # 计算精度、AUC、kappa系数
        result_dict['accuracy'] = np.sum(predictions == Y) / len(Y)
        result_dict['auc'] = self.get_auc(Y, np.array(probas),dt_model.classes_)
        result_dict['kappa'] = cohen_kappa_score(Y,predictions)

        result_dict['n_nodes'] = dt_model.tree_.node_count
        result_dict['probas'] = probas
        return result_dict
    
    def cmm_tree_measures(self,X,Y,dt_model):
        return {k.replace('decision_tree','cmm_tree'):v for k,v in self.decision_tree_measures(X,Y,dt_model).items()}
    def new_model_measures(self,X,Y,new_model,branches_df):
        result_dict={}
        probas,depths=[],[]
        for inst in X:
            prob,depth=new_model.predict_probas_and_depth(inst,branches_df)
            probas.append(prob)
            depths.append(depth)
        predictions=[self.classes_[i] for i in np.array([np.argmax(prob) for prob in probas])]
        result_dict['average_depth']=np.mean(depths)
        result_dict['min_depth'] = np.min(depths)
        result_dict['max_depth'] = np.max(depths)
        result_dict['accuracy'] = np.sum(predictions==Y) / len(Y)
        result_dict['auc'] = self.get_auc(Y,np.array(probas),self.classes_)
        result_dict['kappa'] = cohen_kappa_score(Y,predictions)
        result_dict['n_nodes'] = new_model.number_of_children()
        result_dict['probas'] = probas

        return result_dict
    def ensemble_measures(self,X,Y,rf):
        result_dict={}
        predictions,depths=self.ensemble_prediction(X,rf)
        result_dict['average_depth']=np.mean(depths)
        result_dict['min_depth'] = np.min(depths)
        result_dict['max_depth'] = np.max(depths)
        ensemble_probas=rf.predict_proba(X)
        result_dict['accuracy'] = np.sum(rf.predict(X)==Y)/len(Y)
        result_dict['auc'] = self.get_auc(Y,ensemble_probas,rf.classes_)
        result_dict['kappa'] = cohen_kappa_score(Y,rf.predict(X))
        result_dict['n_nodes'] = sum([t.tree_.node_count for t in rf.estimators_])
        result_dict['probas'] = ensemble_probas
        return result_dict
    def ensemble_prediction(self,X, rf):
        predictions = []
        depths = []
        for inst in X:
            pred = []
            depth = 0
            for base_model in rf.estimators_:
                res = self.tree_depth_and_prediction(inst, base_model.tree_)
                pred.append(res[0])
                depth += res[1]
            predictions.append(np.array(pred).mean(axis=0))
            depths.append(depth)
        return predictions, depths
    def tree_depth_and_prediction(self,inst, t):
        indx = 0
        depth = 0
        epsilon = 0.0000001
        # epsilon: thresholds may be shifted by a very small floating points. For example: x1 <= 2.6 may become x1 <= 2.5999999
        # and then x1 = 2.6 won't be captured
        while t.feature[indx] >= 0:
            if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
                indx = t.children_left[indx]
            else:
                indx = t.children_right[indx]
            depth += 1
        return np.array([i / np.sum(t.value[indx][0]) for i in t.value[indx][0]]), depth
    def get_auc(self,Y,y_score,classes):
        y_test_binarize=np.array([[1 if i ==c else 0 for c in classes] for i in Y])
        fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
        return auc(fpr, tpr)

    def fit_decision_tree_model(self,train_x, train_y):
        """
        This function gets train data and conducts a gridsearch for the best decision tree
        out of several options. It returns the fitted tree
        """
        parameters = {'max_depth': [3, 10, 20],
                      'criterion':['gini','entropy'],
                      'min_samples_leaf': [1, 2, 10]}
        model = DecisionTreeClassifier()
        clfGS = GridSearchCV(model, parameters, cv=10)
        clfGS.fit(train_x, train_y)
        model = clfGS.best_estimator_
        return model
    def fit_cmm_tree(self, df,x_columns,y_column, rf):
        synthetic_data = get_synthetic_data(df)
        cmm_dt = train_dt_for_synthetic_data(synthetic_data,x_columns,y_column,rf)
        return cmm_dt

