import numpy as np
import pandas as pd

import jpype.imports
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from evaluate import count_accuracy
from data import RT, WDAG, SEM
from method import mi_tester, mi_tester_est, pearson, PC_Tree

import pytetrad.tools.TetradSearch as search

try:
    jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
except OSError:
    print("JVM already started")

import config
p = config.setup()
lgr = p.logger

def SHD(true_ud, est_ud):
    return np.sum(np.abs( np.tril(true_ud) - np.tril(est_ud)))

sample = np.array([1000, 2000, 3000, 4000, 5000])
repeat = 50

for i in range(len(sample)):
    p.s = sample[i]

    for j in range(repeat):
        B = RT(p)        # Binary adjacency matrix of random tree
        A = WDAG(B, p)   # weighted DAG
        X = SEM(A, p)    # Data generated by SEM
        A_ud = B + B.T - np.diag(np.diag(B))  # The skeleton of the DAG
        """ Chow-Liu algorithm """
        T_est_2 = mi_tester_est(X)
        fdr_mi2, tpr_mi2, fpr_mi2, shd_mi2, pred_size = count_accuracy(A_ud, T_est_2)
        """ PC-tree algorithm """
        pc_tree = PC_Tree(X, cut=0.05)
        fdr_pctree, tpr_pctree, fpr_pctree, shd_pctree, pred_size = count_accuracy(A_ud, pearon_T)

        """ PC algorithm - from causallearn"""
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz, d_separation
        cg = pc(X, 0.001, fisherz)
        fdr_pc, tpr_pc, fpr_pc, shd_pc, pred_size_pc = count_accuracy(A_ud, np.abs(cg.G.graph))

        """ FCI algorithm - from causallearn"""
        from causallearn.search.ConstraintBased.FCI import fci
        G_1, edges_1 = fci(X, fisherz, 0.001, verbose=False)
        for (i, j) in np.transpose(np.nonzero(G_1.graph)):
            G_1.graph[i, j] = 1
        fdr_fci_1, tpr_fci_1, fpr_fci_1, shd_fci_1, pred_size_fci_1 = count_accuracy(A_ud, G_1.graph)
        """ GES algorithm -  from py-Tetrad"""
        data = pd.DataFrame(X)
        data = data.astype({col: "float64" for col in data.columns})
        search_ges = search.TetradSearch(data)
        search_ges.set_verbose(False)
        search_ges.use_sem_bic()
        search_ges.use_fisher_z(alpha=0.05)
        search_ges.run_fges()
        G_fges = search_ges.get_pcalg().to_numpy()
        G_fges = G_fges + G_fges.T
        for (i, j) in np.transpose(np.nonzero(G_fges)):
            G_fges[i, j] = 1
        fdr_ges, tpr_ges, fpr_ges, shd_ges, pred_size_ges = count_accuracy(A_ud, G_fges)
