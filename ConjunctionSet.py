from Branch import Branch
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import entropy
from pruningFunctions import *
import random


class ConjunctionSet():
    def __init__(self, progress, feature_names, original_data, pruning_x, pruning_y,
                 model, feature_types, amount_of_branches_threshold, filter_approach='probability',
                 exclusion_starting_point=5,
                 minimal_forest_size=10, exclusion_threshold=0.8):
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.model = model
        self.feature_names = feature_names
        self.exclusion_threshold = exclusion_threshold
        self.label_names = model.classes_
        self.relevant_indexes = reduce_error_pruning(self.model, pruning_x, pruning_y, minimal_forest_size)
        # self.relevant_indexes = [i for i in range(len(self.model.estimators_))]
        self.feature_types = feature_types
        self.filter_approach = filter_approach
        self.exclusion_starting_point = exclusion_starting_point
        self.set_ecdf(original_data)
        self.get_ranges(original_data)
        self.generateBranches(progress)
        # self.get_associative_leaves(np.concatenate((original_data,pruning_x)))
        self.number_of_branches_per_iteration = []
        self.buildConjunctionSet(progress)

    def generateBranches(self, progress):
        trees = [estimator.tree_ for estimator in self.model.estimators_]

        # 对于生成分支的过程，是非常快的，所以不需要进度条
        # task_generate_branches = progress.add_task("[red]Generating branches for RF...", total=len(trees))
        self.branches_lists = []
        for i, tree_ in enumerate(trees):
            if i in self.relevant_indexes:
                self.branches_lists.append(self.get_tree_branches(tree_))
                # progress.update(task_generate_branches, advance=1)
            # 一般tree_relevant_num 只有0.1
            # print('tree_relevant_num', len(self.relevant_indexes)/len(trees))
        # progress.stop_task(task_generate_branches)
        for list_indx, branch_list in enumerate(self.branches_lists):  # 枚举分支列表，每个分支列表由一棵树生成
            for leaf_index, branch in enumerate(branch_list):  # 枚举分支列表中的分支
                branch.leaves_indexes = [str(list_indx) + '_' + str(leaf_index)]  # 叶子索引名称是列表序号+叶子序号

    def get_tree_branches(self, tree_):  # 先求出树的所有叶节点，再根据每个叶节点回溯到根节点，生成分支，汇总成一个分支列表
        leaf_indexes = [i for i in range(tree_.node_count) if
                        tree_.children_left[i] == -1 and tree_.children_right[i] == -1]  # 找出所有叶子结点的代码
        branches = [self.get_branch_from_leaf_index(tree_, leaf_index) for leaf_index in leaf_indexes]
        return branches

    def get_branch_from_leaf_index(self, tree_, leaf_index):  # 从叶子结点的序号回溯到根节点，生成分支的函数
        sum_of_probas = np.sum(tree_.value[leaf_index][0])  # 是对每个叶子结点的每个类别数求和
        label_probas = [i / sum_of_probas for i in tree_.value[leaf_index][0]]  # 生成每个类别的概率列表
        new_branch = Branch(self.feature_names, self.feature_types, self.label_names, label_probas=label_probas,
                            number_of_samples=tree_.n_node_samples[leaf_index])  # initialize branch
        node_id = leaf_index
        while node_id:  # iterate over all nodes in branch
            ancestor_index = np.where(tree_.children_left == node_id)[ # np.where()的返回值是一个array，所以要加[0]
                0]  # assuming left is the default for efficiency purposes
            bound = 'upper'  # 如果某个节点的左子节点是node的话，那么就是设定了上界
            if len(ancestor_index) == 0:  # 如果在ancestor中都没找到的话就代表某个节点的右子节点是node
                bound = 'lower'
                ancestor_index = np.where(tree_.children_right == node_id)[0]
            new_branch.addCondition(tree_.feature[ancestor_index[0]], tree_.threshold[ancestor_index[0]],
                                    bound)  # 根据特征和阈值添加一个条件
            node_id = ancestor_index[0]  # 从子节点往根节点找
        return new_branch

    def buildConjunctionSet(self, progress):  # 注意区分连接集和分支集，连接集是分支被合并后的集合
        conjunctionSet = self.branches_lists[0]  # 初始化连接集，只含第一棵树的分支列表
        task_build_conjunction_set = progress.add_task("[red]Building conjunction set for RF...", total=len(self.branches_lists[0:]))
        progress.update(task_build_conjunction_set, advance=1)
        excluded_branches = []
        for i, branch_list in enumerate(self.branches_lists[1:]):
            print('Iteration ' + str(i + 1) + ": " + str(len(conjunctionSet)) + " conjunctions")
            filter = False if i == len(self.branches_lists[1:]) else True # 最后一次的时候不过滤得剩固定数量的分支
            conjunctionSet = self.merge_branch_with_conjunctionSet(branch_list, conjunctionSet, filter=filter)
            # print('i='+str(i))
            if i >= self.exclusion_starting_point and len(conjunctionSet) > 0.8 * self.amount_of_branches_threshold:
                conjunctionSet, this_iteration_exclusions = self.exclude_branches_from_cs(conjunctionSet,
                                                                                          self.exclusion_threshold)
                excluded_branches.extend(this_iteration_exclusions)
            progress.update(task_build_conjunction_set, advance=1)

        self.conjunctionSet = excluded_branches + conjunctionSet
        print('Final CS size: ' + str(len(self.conjunctionSet)))

    def exclude_branches_from_cs(self, cs, threshold): # 从连接集中排除一些分支
        filtered_cs = []
        excludable_branches = []
        for branch in cs:
            if branch.is_excludable_branch(threshold):
                excludable_branches.append(branch)
            else:
                filtered_cs.append(branch)
        return filtered_cs, excludable_branches

    def filter_conjunction_set(self, cs): # 过滤连接集，根据不同的过滤方法
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        if self.filter_approach == 'association': # 根据关联性过滤
            cs = [b for b in cs if b.is_valid_association(self.association_leaves)]
            return cs
        if self.filter_approach == 'probability': # 根据概率过滤，而且连续性数值的范围概率是根据概率累积分布函数来计算的
            branches_metrics = [b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        elif self.filter_approach == 'number_of_samples': # 根据样本数过滤
            branches_metrics = [b.number_of_samples for b in cs]
        elif self.filter_approach == 'probability_entropy': # 根据 分支概率和分支的熵 两个条件相乘来过滤
            branches_metrics = [b.calculate_branch_probability_by_ecdf(self.ecdf_dict) * (1 - entropy(b.label_probas))
                                for b in cs]
        elif self.filter_approach == 'entropy': # 根据熵过滤，熵越大就代表越不确定，越不确定就越不好，所以给entropy加负号了
            branches_metrics = [-entropy(b.label_probas) for b in cs] 
        elif self.filter_approach == 'range': # 通过均匀分布来求概率来过滤
            branches_metrics = [b.calculate_branch_probability_by_range(self.ranges) for b in cs]
        elif self.filter_approach == 'association_probability': # 先判断分支是否是有效的关联分支，然后再计算分支的概率
            branches_metrics = [ b.is_valid_association(self.association_leaves)
                                * b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        threshold = sorted(branches_metrics, reverse=True)[self.amount_of_branches_threshold - 1] # 找出最后一位的分支概率，作为阈值
        return [b for b, metric in zip(cs, branches_metrics) if metric >= threshold][:self.amount_of_branches_threshold] # 如果最后的多个分支概率相同，也不多取

    def merge_branch_with_conjunctionSet(self, branch_list, conjunctionSet, filter=True): # 将分支和连接集合并
        new_conjunction_set = []
        for b1 in conjunctionSet:
            new_conjunction_set.extend([b1.mergeBranch(b2) for b2 in branch_list if b1.contradictBranch(b2) == False]) # 判断分支是否矛盾，不矛盾的话合并
        # print('number of branches before filterring: '+str(len(new_conjunction_set)))
        if filter:
            new_conjunction_set = self.filter_conjunction_set(new_conjunction_set)
        # print('number of branches after filterring: ' + str(len(new_conjunction_set)))
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set

    def get_conjunction_set_df(self): # 将连接集转换为DataFrame
        return pd.DataFrame([b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet])

    def predict(self, X):
        predictions = []
        for inst in X:
            for conjunction in self.conjunctionSet:
                if conjunction.containsInstance(inst):
                    predictions.append(self.label_names[conjunction.getLabel()])
        return predictions

    def get_instance_branch(self, inst):
        for conjunction in self.conjunctionSet:
            if conjunction.containsInstance(inst):
                return conjunction

    def set_ecdf(self, data): # ECDF是经验累积分布函数，是一种非参数的统计方法，这里对训练集中的每个特征都计算了一个ECDF
        self.ecdf_dict = {i: ECDF(data.transpose().T[i]) for i in range(len(self.feature_names))} # data.transpose().T[i]是获取第i列的数据，即第i个特征的数据，然后计算它的ECDF

    def group_by_label_probas(self, conjunctionSet): # 将所有结果为同一类的branch分在一起
        probas_hashes = {}
        for i, b in enumerate(conjunctionSet):
            probas_hash = hash(tuple(b.label_probas)) # 只要是相同的东西，hash值一定是一样的
            if probas_hash not in probas_hashes:
                probas_hashes[probas_hash] = []
            probas_hashes[probas_hash].append(i)
        return probas_hashes

    def get_ranges(self, original_data): # 为每个特征计算取值范围
        self.ranges = [max(v) - min(v) for v in original_data.transpose()]

    def get_associative_leaves(self, X): # 对不同树上的叶子计算关联性
        association_dict = {}
        for indx, inst in enumerate(X):
            association_dict[indx] = []
            for tree_indx, tree_ in enumerate(self.branches_lists):
                for leaf_indx, leaf in enumerate(tree_):
                    if leaf.containsInstance(inst):
                        association_dict[indx].append(str(tree_indx) + '_' + str(leaf_indx))
        association_dict
        self.association_leaves = {}
        for tree_indx1, tree_1 in enumerate(self.branches_lists):# 先将所有不同树上的两个叶节点的关联度初始化为0
            for tree_indx2, tree_2 in enumerate(self.branches_lists):
                if tree_indx1 == tree_indx2:
                    continue
                for leaf_index1, leaf1 in enumerate(tree_1): 
                    for leaf_index2, leaf2 in enumerate(tree_2):
                        self.association_leaves[
                            str(tree_indx1) + '_' + str(leaf_index1) + '|' + str(tree_indx2) + '_' + str(
                                leaf_index2)] = 0
        for inst in association_dict: # 
            for leaf1 in association_dict[inst]:
                for leaf2 in association_dict[inst]:
                    if leaf1 == leaf2:
                        continue
                    else:
                        self.association_leaves[leaf1 + '|' + leaf2] += 1
