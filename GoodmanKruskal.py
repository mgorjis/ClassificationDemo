import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


EPS = np.finfo(float).eps

def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes

def goodman_kruskal_tau(ref_labels, sys_labels, cm=None):
    """Return Goodman-Kruskal tau between ``ref_labels`` and ``sys_labels``.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    tau_ref_sys : float
        Value between 0 and 1 that is high when ``ref_labels`` is predictive
        of ``sys_labels`` and low when ``ref_labels`` provides essentially no
        information about ``sys_labels``.
    tau_sys_ref : float
        Value between 0 and 1 that is high when ``sys_labels`` is predictive
        of ``ref_labels`` and low when ``sys_labels`` provides essentially no
        information about ``ref_labels``.
    References
    ----------
    - Goodman, L.A. and Kruskal, W.H. (1954). "Measures of association for
      cross classifications." Journal of the American Statistical Association.
    - Pearson, R. (2016). GoodmanKruskal: Association Analysis for Categorical
      Variables. https://CRAN.R-project.org/package=GoodmanKruskal.
    """
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm = cm / cm.sum()
    ref_marginals = cm.sum(axis=1)
    sys_marginals = cm.sum(axis=0)

    # Tau(ref, sys).
    vy = 1 - np.sum(sys_marginals**2) + EPS
    xy_term = np.sum(cm**2, axis=1)
    vy_bar_x = 1 - np.sum(xy_term / ref_marginals)
    tau_ref_sys = (vy - vy_bar_x) / vy

    # Tau(sys, ref).
    vx = 1 - np.sum(ref_marginals**2) + EPS
    yx_term = np.sum(cm**2, axis=0)
    vx_bar_y = 1 - np.sum(yx_term / sys_marginals)
    tau_sys_ref = (vx - vx_bar_y) / vx

    return tau_ref_sys, tau_sys_ref


def GKtauDataframe(df):

    df_matrix= df.values
    columns = df.columns
    n= len(columns)
    GK_matrix = np.zeros([n,n])

    for i in range(0,n):
        for j in range(0,n):

            gk = goodman_kruskal_tau(df_matrix[:,i], df_matrix[:,j])

            GK_matrix[i,j]= gk[0]
            GK_matrix[j,i]= gk[1]

    GK_Dataframe = pd.DataFrame(GK_matrix)

    GK_Dataframe.columns = columns
    GK_Dataframe.index= columns
    return(GK_Dataframe)




def display_correlation(corr, upper ):

    n= corr.shape[0]
    if upper == False:
        keep = (np.ones([n,n]))
        np.fill_diagonal(keep, 0)
        keep = keep.astype('bool').reshape(corr.size)
    else:
        keep = np.triu(np.ones(corr.shape),1).astype('bool').reshape(corr.size)

    corr_stack = corr.stack()[keep].reset_index()
    corr_stack.columns = ['x1','x2','correlation'] 
    
    return corr_stack.sort_values("correlation",ascending = False)




# #make the diagonal 0
#     def pd_fill_diagonal(df_matrix, value=0): 
#         mat = df_matrix.values
#         n = mat.shape[0]
#         mat[range(n), range(n)] = value
#         return pd.DataFrame(mat)

#     j = corr.where(corr>threshold)
#     j = pd_fill_diagonal(j, 0) 
#     k= np.where(j>threshold)
#     m= list(zip(  k[0],k[1]))

#     columns = corr.columns
#     all = []
#     for i in m:
#         all.append([columns[i[0]],columns[i[1]],j.loc[i[0],i[1]]])

#     all = pd.DataFrame(all)
    
#     if len(all)==0:
#         all = pd.DataFrame([[],[],[]]).transpose()
    
#     all.columns = ['x1','x2','correlation'] 
#     all.sort_values('correlation',ascending = False)
    
#     return all.sort_values('correlation',ascending = False)