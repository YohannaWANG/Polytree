import numpy as np
import networkx as nx
import config
p = config.setup()
lgr = p.logger

n = p.n
def mutual_info_est(x, y):
    """
    Estimate coefficient beta use OLS
    """
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    beta = float(reg.coef_)

    z = np.stack((x, y), axis=0)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.cov(z)
    det_xy = np.linalg.det(sigma_xy)

    I_xy = 1 / 2 * np.log(1 + np.square(beta)*np.square(sigma_x)/np.square(sigma_y))
    return I_xy
def mutual_info(x, y):
    """
    Mutual information for a continuous target variable.
    """
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(y.reshape(-1,1), x.reshape(-1,1))
    return mi

def pearson(X):
    from scipy.stats import pearsonr
    Corr = np.identity(n)
    for i in range(n):
        for j in range(n):
            corr, _ = pearsonr(X[:,i], X[:,j])
            Corr[i,j] = corr
    G_corr= nx.from_numpy_array(np.abs(Corr))
    T_est = nx.maximum_spanning_tree(G_corr)
    return nx.to_numpy_array(T_est)

def mi_tester_est(X):
    Mi = np.identity(n)
    for i in range(n):
        for j in range(n):
            mi = mutual_info_est(X[:,i], X[:,j])
            Mi[i,j] = mi
    G_mi= nx.from_numpy_array(np.abs(Mi))
    T_est = nx.maximum_spanning_tree(G_mi)
    return nx.to_numpy_array(T_est)

def mi_tester(X):
    Mi = np.identity(n)
    for i in range(n):
        for j in range(n):
            mi = mutual_info(X[:,i], X[:,j])
            Mi[i,j] = mi
    G_mi= nx.from_numpy_array(np.abs(Mi))
    T_est = nx.maximum_spanning_tree(G_mi)
    return nx.to_numpy_array(T_est)

"""
Function: PC tree algorithm. 
"""
def PC_Tree(X, cut=0.05):
    '''
    PC tree algorithm for learning faithful polytree

    Arguments:
        X: (n, d) data matrix
        cut: CI test threshold
    Returns:
        numpy.ndarray: (d, d) symmetric matrix of skeleton.
    '''
    n, d = X.shape
    skeleton = np.zeros((d,d))
    Sigma = X.T @ X / n

    def _mItest(j,k):
        '''
        Marginal independence test
        return True if accept independence
        '''
        if abs(Sigma[j,k]) / np.sqrt(Sigma[j,j] * Sigma[k,k]) > cut:
            return False
        else:
            return True

    def _CItest(j,k,l):
        '''
        Conditional independence test
        return True if accept independence
        '''
        num = Sigma[j,k] - Sigma[j,l] / Sigma[l,l] * Sigma[l,k]

        den1 = Sigma[j,j] - Sigma[j,l] / Sigma[l,l] * Sigma[l,j]
        den2 = Sigma[k,k] - Sigma[k,l] / Sigma[l,l] * Sigma[l,k]

        if abs(num) / np.sqrt(den1 * den2) > cut:
            return False
        else:
            return True

    for j in range(d):
        for k in range(j+1, d):
            d_spr_flag = 0
            if _mItest(j,k):
                d_spr_flag = 1
            else:
                for l in range(d):
                    if l != j and l != k:
                        if _CItest(j,k,l):
                            d_spr_flag = 1
                            break

            if d_spr_flag != 1:
                skeleton[j,k], skeleton[k,j] = 1, 1

    return skeleton