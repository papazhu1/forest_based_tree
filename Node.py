import numpy as np
import pickle
from SplittingFunctions import *
from scipy.stats import entropy
EPSILON=0.000001
class Node():
    def __init__(self,mask): # 在决策树新建的时候，传入了长度为 conjunction 个数的全是true的列表
        self.mask = mask # seLf.mask表示当前还在这个节点下的conjunction，是一个boolean list
    def split(self,df): # 两个终止条件，一个是只剩下一个conjunction了，另一个是 not is_splitable
        #if np.sum(self.mask)==1 or self.has_same_class(df):
        if np.sum(self.mask) == 1: # mask中只剩一个true的时候，说明就只剩下一条规则了，因此这个节点是叶子节点
            self.left=None
            self.right=None
            return
        self.features = [int(i.split('_')[0]) for i in df.keys() if 'upper' in str(i)] # 当前的conjunction中特征全是以 ***_upper、 ***_lower的形式存在的,split就可以获得特征名

        self.split_feature, self.split_value = self.select_split_feature(df)
        self.create_mask(df)
        is_splitable=self.is_splitable()
        if is_splitable==False:
            self.left = None
            self.right = None
            return
        
        # 如果划分在某个conjunction的中间，则该conjunction会被分到左右子树中，因为确实两边都符合conjunction的条件
        self.left=Node(list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask))))
        self.right = Node(list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask))))
        self.left.split(df)
        self.right.split(df)

    def is_splitable(self):
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == 0 or np.sum(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == 0:
            return False
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == np.sum(self.mask) or np.sum(
            np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == np.sum(self.mask):
            return False
        return True

    def create_mask(self,df): # 判断每个conjuntion划分到了左子树还是右子树，或者划分值在conjunction该特征的区间内，有别于self.mask
        # 每个结果都类似为[True,False,False,True,False,True,False,False], 是一个 boolean list 
        self.left_mask = df[str(self.split_feature) + "_upper"] <= self.split_value 
        self.right_mask = df[str(self.split_feature) + '_lower'] >= self.split_value
        self.both_mask = ((df[str(self.split_feature) + '_lower'] < self.split_value) & (df[str(self.split_feature) + "_upper"] > self.split_value))
        #self.both_mask = [True if self.split_value < upper and self.split_value > lower else False for lower, upper in
        #             zip(df[str(self.split_feature) + '_lower'], df[str(self.split_feature) + "_upper"])]

    def select_split_feature(self,df):
        feature_to_value={}
        feature_to_metric={}
        for feature in self.features: # 对每个特征都试一下划分，看看哪个特征的分裂效果最好
           value,metric=self.check_feature_split_value(df,feature) # 对当前被划分特征选出最好的划分点
           feature_to_value[feature] = value
           feature_to_metric[feature] = metric
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature,feature_to_value[feature]

    def check_feature_split_value(self,df,feature): # 对当前遍历的划分特征选出最好的划分点函数
        value_to_metric={} # value_to_metric是一个字典，包含三个划分点，以及每个划分点的metric
        values=list(set(list(df[str(feature)+'_upper'][self.mask])+list(df[str(feature)+'_lower'][self.mask]))) # 索引类型是boolean arrays，找出当前还是true的值
        np.random.shuffle(values)
        values=values[:3] # 那意思就是随机抽取三个划分点？？？
        for value in values:
            left_mask=[True if upper <= value  else False for upper in df[str(feature)+"_upper"]]
            right_mask=[True if lower>= value else False for lower in df[str(feature)+'_lower']]
            both_mask=[True if value < upper and value> lower else False for lower,upper in zip(df[str(feature)+'_lower'],df[str(feature)+"_upper"])]
            value_to_metric[value]=self.get_value_metric(df,left_mask,right_mask,both_mask)
        val=min(value_to_metric,key=value_to_metric.get)
        return val,value_to_metric[val]

    def get_value_metric(self,df,left_mask,right_mask,both_mask): # 计算划分点的评分
        l_df_mask=np.logical_and(np.logical_or(left_mask,both_mask),self.mask) # 利用逻辑或，将分到左子树和分到both的都选出来，再挑出还在当前节点下的
        r_df_mask=np.logical_and(np.logical_or(right_mask,both_mask),self.mask) # 利用逻辑或，将分到右子树和分到both的都选出来，再挑出还在当前节点下的
        if np.sum(l_df_mask)==0 or np.sum(r_df_mask)==0: # 如果左子树或者右子树为空，则返回无穷大
            return np.inf
        l_entropy,r_entropy=self.calculate_entropy(df,l_df_mask),self.calculate_entropy(df,r_df_mask) # 对所有conjunction同等对待，求所有类的平均值，然后求熵
        l_prop=np.sum(l_df_mask) / len(l_df_mask)
        # r_prop=np.sum(r_df_mask) / len(l_df_mask) # 这里写错了吧，应该是len(r_df_mask)吧
        r_prop = np.sum(r_df_mask) / len(r_df_mask)
        return l_entropy * l_prop + r_entropy * r_prop

    def predict_probas_and_depth(self,inst,training_df):
        if self.left is None and self.right is None:
            return self.node_probas(training_df),1
        if inst[self.split_feature] <= self.split_value:
            prediction,depth = self.left.predict_probas_and_depth(inst,training_df)
            return prediction,depth + 1
        else:
            prediction, depth = self.right.predict_probas_and_depth(inst, training_df)
            return prediction, depth + 1

    def node_probas(self, df):
        x = df['probas'][self.mask].mean()
        return x/x.sum()
    def get_node_prediction(self,training_df):
        v=training_df['probas'][self.mask][0]
        v=[i/np.sum(v) for i in v]
        return np.array(v)
    def opposite_col(self,s):
        if 'upper' in s:
            return s.replace('upper','lower')
        else:
            return s.replace('lower', 'upper')
    def calculate_entropy(self,test_df,test_df_mask):
        x = test_df['probas'][test_df_mask].mean() # 对conjunction的概率求均值，对每个conjunction的概率同等对待，没有因为conjunction包含的样本数印象
        return entropy(x/x.sum())
    def count_depth(self):
        if self.right==None:
            return 1
        return max(self.left.count_depth(),self.right.count_depth())+1
    def number_of_children(self):
        if self.right==None:
            return 1
        return 1+self.right.number_of_children()+self.left.number_of_children()
    def has_same_class(self,df):
        labels=set([np.argmax(l) for l in df['probas'][self.mask]])
        if len(labels)>1:
            return False
        return True