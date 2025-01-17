import numpy as np
from operator import mul
from operator import mul
from functools import reduce
EPSILON=0.001
def get_prob(i,features_upper,features_lower,ecdf):
    return ecdf[i]([features_lower[i], features_upper[i]])

class Branch:
    def __init__(self,feature_names,feature_types,label_names,label_probas=None,number_of_samples=None):
        """Branch inatance can be initialized in 2 ways. One option is to initialize an empty branch
        (only with a global number of features and number of class labels) and gradually add
        conditions - this option is relevant for the merge implementation.
        Second option is to get the number of samples in branch and the labels
        probability vector - relevant for creating a branch out of an existing tree leaf.
        """
        self.feature_types=feature_types # 特征类型列表
        self.label_names=label_names # 标签名称列表
        self.number_of_features=len(feature_names) # 特征数量
        self.feature_names=feature_names # 特征名称列表
        self.features_upper=[np.inf]*self.number_of_features #upper bound of the feature for the given rule
        self.features_lower=[-np.inf]*self.number_of_features #lower bound of the feature for the given rule
        self.label_probas=label_probas # 
        # 作者没有用到分支下样本数量，因为工作是构建conjunction的决策树，因此将conjunction同等对待，而conjunction下样本的多少是没有意义的save number of samples in leaf (not relevant for the current model) 
        self.number_of_samples=number_of_samples 
        self.categorical_features_dict={}
    def addCondition(self, feature, threshold, bound): # 这个称为changeCondition更合适吧，因为是更新上下界的阈值，而不是添加新的条件
        """
        This function gets feature index, its threshold for the condition and whether
        it is upper or lower bound. It updates the features thresholds for the given rule.
        """
        if bound == 'lower':
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = threshold
                if '=' in self.feature_names[feature] and threshold >= 0: # 如果特征名如“水果=苹果”，则拆分为字典形式
                    splitted = self.feature_names[feature].split('=')
                    self.categorical_features_dict[splitted[0]]=splitted[1]
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = threshold
    def contradictBranch(self, other_branch):
        """
        check weather Branch b can be merged with the "self" Branch. Returns Boolean answer.
        """
        for categorical_feature in self.categorical_features_dict: # 类别特征不相等则矛盾
            if categorical_feature in other_branch.categorical_features_dict and self.categorical_features_dict[categorical_feature] != other_branch.categorical_features_dict[categorical_feature]:
                return True
        for i in range(self.number_of_features):
            if self.features_upper[i] <= other_branch.features_lower[i] + EPSILON or self.features_lower[i] + EPSILON >= other_branch.features_upper[i]:
                return True
            # 如果自己这个特征类型已经是整数了，如果另一个分支的上界比自己的上界小，且另一个分支的上界是小数，且两个分支的上界差值小于1，则矛盾
            # 比如[1,2]和[1,1.5],也是没有意义的
            if self.feature_types[i] == 'int' and min(self.features_upper[i],other_branch.features_upper[i])%1>0 and \
                                    min(self.features_upper[i],other_branch.features_upper[i])-max(self.features_lower[i],other_branch.features_lower[i])<1:
                return True

        return False
    def mergeBranch(self, other_branch): # 这里只是固定将确定的两个分支合并
        """
        This method gets Branch b and create a new branch which is a merge of the "self" object
        with b. As describe in the algorithm.
        """
        new_label_probas=[k+v for k,v in zip(self.label_probas,other_branch.label_probas)] # 概率直接相加，因为这里的概率只是为了算类别占比
        new_number_of_samples=np.sqrt(self.number_of_samples * other_branch.number_of_samples) # 启发式算法根号下乘积，而直接相加可能会重复相加
        new_b = Branch(self.feature_names,self.feature_types,self.label_names,new_label_probas,new_number_of_samples)
        new_b.features_upper, new_b.features_lower = list(self.features_upper), list(self.features_lower)
        for feature in range(self.number_of_features): # 遍历更新特征上下界
            new_b.addCondition(feature, other_branch.features_upper[feature], 'upper')
            new_b.addCondition(feature, other_branch.features_lower[feature], 'lower')
        new_b.categorical_features_dict = dict(self.categorical_features_dict)
        new_b.categorical_features_dict.update(dict(other_branch.categorical_features_dict))
        new_b.leaves_indexes = self.leaves_indexes + other_branch.leaves_indexes
        return new_b
    def toString(self):
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = ""
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                #s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + ' > ' + str(np.round(threshold, 3)) + ", "
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                #s +=  self.feature_names[feature] + ' <= ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + ' <= ' + str(np.round(threshold, 3)) + ", "
        s += 'labels: ['
        for k in range(len(self.label_probas)):
            s+=str(self.label_names[k])+' : '+str(self.label_probas[k])+' '
        s+=']'
        s+=' Number of samples: '+str(self.number_of_samples)
        return s
    def printBranch(self):
        # print the branch by using tostring()
        print(self.toString())


    def containsInstance(self, instance):
        """This function gets an ibservation as an input. It returns True if the set of rules
        that represented by the branch matches the instance and false otherwise.
        """
        # 把所有true加起来，如果等于特征数量，则说明所有特征都满足条件
        if np.sum(self.features_upper >= instance)==len(instance) and np.sum(self.features_lower < instance)==len(instance): 
            return True
        return False
    def getLabel(self): # 对一个分支，返回概率最大的类别
        # Return the predicted label according to the branch
        return np.argmax(self.label_probas)
    def containsInstance(self, v):
        for i,lower,upper in zip(range(len(v)),self.features_lower,self.features_upper):
            if v[i]>upper or v[i]<=lower:
                return False
        return True
    def get_branch_dict(self,ecdf): # 将分支转换为字典形式
        features={}
        for feature,upper_value,lower_value in zip(range(len(self.features_upper)),self.features_upper,self.features_lower):
            features[str(feature)+'_upper']=upper_value
            features[str(feature)+'_lower']=lower_value
        features['number_of_samples']=self.number_of_samples
        features['branch_probability'] = self.calculate_branch_probability_by_ecdf(ecdf)
        features['probas']=np.array(self.label_probas)
        return  features

    # 计算这个分支的概率，这个概率是根据每个特征的累积概率分布函数计算概率的乘积计算的
    def calculate_branch_probability_by_ecdf(self, ecdf): # ecdf 是一个函数列表，它包含了每个特征的 ECDF 函数。这些 ECDF 函数描述了每个特征的值在数据集中的累积概率分布,前提条件是所有的特征都是独立的。
        features_probabilities=[] # ECDF在ConjunctionSet中定义
        delta = 0.000000001 # 防止概率为0
        for i in range(len(ecdf)):
            probs=ecdf[i]([self.features_lower[i],self.features_upper[i]])
            features_probabilities.append((probs[1]-probs[0]+delta)) 
        return np.product(features_probabilities) # np.product()是求列表元素的乘积
    
    def calculate_branch_probability_by_range(self, ranges): # 根据范围计算分支概率，全部当成均匀分布了，好像没有被用到
        features_probabilities = 1
        for range, lower, upper in zip(ranges, self.features_lower, self.features_upper):
            probs = min(1,(upper-lower)/range)
        features_probabilities = features_probabilities*probs
        return features_probabilities
    
    def is_excludable_branch(self,threshold): # 判断分支是否可以被排除，如果最大概率大于阈值，则可以被排除，感觉违反常理了
        if max(self.label_probas)/np.sum(self.label_probas)>threshold:
            return True
        return False
    def is_addable(self,other): # 这个函数没有被用到，也是查看是否可以merge的
        for feature in range(self.number_of_features):
            if self.features_upper[feature] + EPSILON < other.features_lower[feature] or other.features_upper[feature] + EPSILON < self.features_lower[feature]:
                return False
        return True
    def is_valid_association(self,associative_leaves): # 判断分支是否是有效的关联分支，使用了一个字典，这个字典是在ConjunctionSet中定义的
        for leaf1 in self.leaves_indexes: # 字典索引是两个叶子的索引，值是0或1，0表示两个叶子之间没有关联，1表示有关联
            for leaf2 in self.leaves_indexes:
                if leaf1 == leaf2:
                    continue
                if associative_leaves[leaf1+'|'+leaf2]==0:
                    return False
        return True
    def number_of_unseen_pairs(self,associative_leaves): # 计算分支中没有关联的叶子对的数量
        count=0
        for leaf1 in self.leaves_indexes:
            for leaf2 in self.leaves_indexes:
                if leaf1 == leaf2:
                    continue
                if associative_leaves[leaf1+'|'+leaf2]==0:
                    count+=1
        return count*(-1)


