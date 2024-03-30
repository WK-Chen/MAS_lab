import numpy as np


class SquareErrorUtils:
    @staticmethod
    def _set_sample_weight(sample_weight, n_samples):
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_samples)
        return sample_weight

    @staticmethod
    def square_error(y, sample_weight):
        y = np.asarray(y)
        return np.sum((y - y.mean()) ** 2 * sample_weight)

    def cond_square_error(self, x, y, sample_weight):
        x, y = np.asarray(x), np.asarray(y)
        error = 0.0
        for x_val in set(x):
            x_idx = np.where(x == x_val)  # 按区域计算误差
            new_y = y[x_idx]  # 对应区域的目标值
            new_sample_weight = sample_weight[x_idx]
            error += self.square_error(new_y, new_sample_weight)
        return error

    def square_error_gain(self, x, y, sample_weight=None):
        sample_weight = self._set_sample_weight(sample_weight, len(x))
        return self.square_error(y, sample_weight) - self.cond_square_error(x, y, sample_weight)


class TreeNode_R:
    def __init__(self, feature_idx: int = None, feature_val=None, y_hat=None, square_error: float = None,
                 criterion_val=None, n_samples: int = None, left_child_Node=None, right_child_Node=None):
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.criterion_val = criterion_val
        self.square_error = square_error
        self.n_samples = n_samples
        self.y_hat = y_hat
        self.left_child_Node = left_child_Node  # 递归
        self.right_child_Node = right_child_Node  # 递归

    def level_order(self):
        pass

class DataBinsWrapper:
    def __init__(self, max_bins=10):
        self.max_bins = max_bins
        self.XrangeMap = None

    def fit(self, x_samples):
        if x_samples.ndim == 1:
            n_features = 1
            x_samples = x_samples[:, np.newaxis]
        else:
            n_features = x_samples.shape[1]

        # 构建分箱，区间段
        self.XrangeMap = [[] for _ in range(n_features)]
        for idx in range(n_features):
            x_sorted = sorted(x_samples[:, idx])
            for bin in range(1, self.max_bins):
                p = (bin / self.max_bins) * 100 // 1
                p_val = np.percentile(x_sorted, p)
                self.XrangeMap[idx].append(p_val)
            self.XrangeMap[idx] = sorted(list(set(self.XrangeMap[idx])))

    def transform(self, x_samples, XrangeMap=None):
        if x_samples.ndim == 1:
            if XrangeMap is not None:
                return np.asarray(np.digitize(x_samples, XrangeMap[0])).reshape(-1)
            else:
                return np.asarray(np.digitize(x_samples, self.XrangeMap[0])).reshape(-1)
        else:
            return np.asarray([np.digitize(x_samples[:, i], self.XrangeMap[i])
                               for i in range(x_samples.shape[1])]).T


class DecisionTreeRegression:
    def __init__(self, criterion="mse", max_depth=None, min_sample_split=2, min_sample_leaf=1,
                 min_target_std=1e-3, min_impurity_decrease=0, max_bins=10):
        self.utils = SquareErrorUtils()
        self.criterion = criterion
        if criterion.lower() == "mse":
            self.criterion_func = self.utils.square_error_gain
        else:
            raise ValueError("参数criterion仅限mse...")
        self.min_target_std = min_target_std
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.root_node: TreeNode_R() = None
        self.dbw = DataBinsWrapper(max_bins=max_bins)
        self.dbw_XrangeMap = {}

    def fit(self, x_train, y_train, sample_weight=None):
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.class_values = np.unique(y_train)
        n_samples, n_features = x_train.shape
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_samples)
        self.root_node = TreeNode_R()
        self.dbw.fit(x_train)
        x_train = self.dbw.transform(x_train)
        self._build_tree(1, self.root_node, x_train, y_train, sample_weight)

    def _build_tree(self, cur_depth, cur_node: TreeNode_R, x_train, y_train, sample_weight):
        n_samples, n_features = x_train.shape
        cur_node.y_hat = np.dot(sample_weight / np.sum(sample_weight), y_train)
        cur_node.n_samples = n_samples

        cur_node.square_error = ((y_train - y_train.mean()) ** 2).sum()
        if cur_node.square_error <= self.min_target_std:
            return
        if n_samples < self.min_sample_split:
            return
        if self.max_depth is not None and cur_depth > self.max_depth:
            return

        best_idx, best_val, best_criterion_val = None, None, 0.0
        for k in range(n_features):
            for f_val in sorted(np.unique(x_train[:, k])):
                region_x = (x_train[:, k] <= f_val).astype(int)
                criterion_val = self.criterion_func(region_x, y_train, sample_weight)
                if criterion_val > best_criterion_val:
                    best_criterion_val = criterion_val
                    best_idx, best_val = k, f_val

        if best_idx is None:
            return
        if best_criterion_val <= self.min_impurity_decrease:
            return
        cur_node.criterion_val = best_criterion_val
        cur_node.feature_idx = best_idx
        cur_node.feature_val = best_val

        left_idx = np.where(x_train[:, best_idx] <= best_val)
        if len(left_idx) >= self.min_sample_leaf:
            left_child_node = TreeNode_R()
            cur_node.left_child_Node = left_child_node
            self._build_tree(cur_depth + 1, left_child_node, x_train[left_idx],
                             y_train[left_idx], sample_weight[left_idx])

        right_idx = np.where(x_train[:, best_idx] > best_val)
        if len(right_idx) >= self.min_sample_leaf:
            right_child_node = TreeNode_R()
            cur_node.right_child_Node = right_child_node
            self._build_tree(cur_depth + 1, right_child_node, x_train[right_idx],
                             y_train[right_idx], sample_weight[right_idx])

    def _search_tree_predict(self, cur_node: TreeNode_R, x_test):
        if cur_node.left_child_Node and x_test[cur_node.feature_idx] <= cur_node.feature_val:
            return self._search_tree_predict(cur_node.left_child_Node, x_test)
        elif cur_node.right_child_Node and x_test[cur_node.feature_idx] > cur_node.feature_val:
            return self._search_tree_predict(cur_node.right_child_Node, x_test)
        else:
            return cur_node.y_hat

    def predict(self, x_test):
        x_test = np.asarray(x_test)
        if self.dbw.XrangeMap is None:
            raise ValueError("请先进行回归决策树的创建，然后预测...")
        x_test = self.dbw.transform(x_test)
        y_test_pred = []
        for i in range(x_test.shape[0]):
            y_test_pred.append(self._search_tree_predict(self.root_node, x_test[i]))
        return np.asarray(y_test_pred)

    @staticmethod
    def cal_mse_r2(y_test, y_pred):
        y_test, y_pred = y_test.reshape(-1), y_pred.reshape(-1)
        mse = ((y_pred - y_test) ** 2).mean()  # 均方误差
        r2 = 1 - ((y_pred - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        return mse, r2

    def _prune_node(self, cur_node: TreeNode_R, alpha):
        if cur_node.left_child_Node:
            self._prune_node(cur_node.left_child_Node, alpha)
        if cur_node.right_child_Node:
            self._prune_node(cur_node.right_child_Node, alpha)

        if cur_node.left_child_Node is not None or cur_node.right_child_Node is not None:
            for child_node in [cur_node.left_child_Node, cur_node.right_child_Node]:
                if child_node is None:
                    continue
                if child_node.left_child_Node is not None or child_node.right_child_Node is not None:
                    return
            pre_prune_value = 2 * alpha
            if cur_node and cur_node.left_child_Node is not None:
                pre_prune_value += (0.0 if cur_node.left_child_Node.square_error is None
                                    else cur_node.left_child_Node.square_error)
            if cur_node and cur_node.right_child_Node is not None:
                pre_prune_value += (0.0 if cur_node.right_child_Node.square_error is None
                                    else cur_node.right_child_Node.square_error)

            after_prune_value = alpha + cur_node.square_error

            if after_prune_value <= pre_prune_value:
                cur_node.left_child_Node = None
                cur_node.right_child_Node = None
                cur_node.feature_idx, cur_node.feature_val = None, None
                cur_node.square_error = None

    def prune(self, alpha=0.01):
        self._prune_node(self.root_node, alpha)
        return self.root_node



if __name__ == '__main__':
    import pickle
    import joblib
    from tqdm import trange
    # from sklearn.model_selection import train_test_split

    with open("./state_records.pkl", 'rb') as f:
        data = pickle.load(f)

    print(f"Data Length: {len(data)}")

    data = np.array(data)

    X = data[:, :-1]
    Y = data[:, -1:]
    print(X.shape)
    print(Y.shape)
    X_train, y_train = X[:-1000], Y[:-1000]
    X_test, y_test = X[-1000:], Y[-1000:]

    tree = DecisionTreeRegression(max_bins=50, max_depth=10)
    for i in trange(30):
        tree.fit(X_train[i*1000:(i+1)*1000], y_train[i*1000:(i+1)*1000])
    y_test_pred = tree.predict(X_test)
    mse, r2 = tree.cal_mse_r2(y_test, y_test_pred)

    print(mse, r2)

    with open('predict_rv_model.pkl', 'wb') as file:
        pickle.dump(tree, file)

    # plt.figure(figsize=(14, 5))
    # plt.subplot(121)
    # plt.scatter(data, target, s=15, c="k", label="Raw Data")
    # plt.plot(x_test, y_test_pred, "r-", lw=1.5, label="Fit Model")
    # plt.xlabel("x", fontdict={"fontsize": 12, "color": "b"})
    # plt.ylabel("y", fontdict={"fontsize": 12, "color": "b"})
    # plt.grid(ls=":")
    # plt.legend(frameon=False)
    # plt.title("Regression Decision Tree(UnPrune) and MSE = %.5f R2 = %.5f" % (mse, r2))
    #
    # plt.subplot(122)
    # tree.prune(0.5)
    # y_test_pred = tree.predict(x_test[:, np.newaxis])
    # mse, r2 = tree.cal_mse_r2(obj_fun(x_test), y_test_pred)
    # plt.scatter(data, target, s=15, c="k", label="Raw Data")
    # plt.plot(x_test, y_test_pred, "r-", lw=1.5, label="Fit Model")
    # plt.xlabel("x", fontdict={"fontsize": 12, "color": "b"})
    # plt.ylabel("y", fontdict={"fontsize": 12, "color": "b"})
    # plt.grid(ls=":")
    # plt.legend(frameon=False)
    # plt.title("Regression Decision Tree(Prune) and MSE = %.5f R2 = %.5f" % (mse, r2))
    #
    # plt.show()

