import os
import json
import numpy as np
import pandas as pd
import GPy

import gurobipy as gp
from gurobipy import GRB

from data_generation import load_data
from utils import *

def min_noise(papers, reviewers, cardinal=False, top_percent=0.1, constr='linear'):
    n_paper = len(papers)
    paper_per = len(reviewers[0]["paper_indices"])
    eps_len = len(reviewers) * paper_per

    try:
        m = gp.Model("minnoise")
        x = m.addMVar(n_paper, name="x")
        eps = m.addMVar(eps_len, lb=-100, name="eps")

    
        m.setObjective(eps @ eps, GRB.MINIMIZE)
        # construct ordinal constraint
        for rev_idx, rev in enumerate(reviewers):
            indices = rev["paper_indices"]
            scores = rev["rev_scores"]

            base_idx = rev_idx * paper_per

            for i in range(len(indices)-1):
                j = i+1

                idx_i, idx_j = indices[i], indices[j]
                assert scores[i] <= scores[j]
                idx_eps_i, idx_eps_j = base_idx + i, base_idx + j

                m.addConstr((x[idx_j] + eps[idx_eps_j]) - (x[idx_i] + eps[idx_eps_i]) >= (scores[j] - scores[i]) / 10)
                

        if cardinal:
            m.addConstr(x <= 1000)
            for rev_idx, rev in enumerate(reviewers):
                pairs = list(zip(rev["paper_indices"], rev["rev_scores"]))
                for i in range(len(pairs)-2):
                    j = i+1
                    k = i+2

                    idx1, y1 = pairs[i]
                    idx2, y2 = pairs[j]
                    idx3, y3 = pairs[k]

                    assert y3 >= y2
                    assert y2 >= y1

                    if y3 == y2 or y2 == y1:
                        continue

                    mul1 = 1 / (y2 - y1 + 0)
                    mul2 = 1 / (y3 - y2 + 0)

                    # convexity constraint
                    base_idx = rev_idx * paper_per

                    idx_eps1 = base_idx + i
                    idx_eps2 = base_idx + j
                    idx_eps3 = base_idx + k

                    
                    if constr == 'linear':
                        m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 == (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                    elif constr == 'convex':
                        m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 >= (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                    elif constr == 'concave':
                        m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 <= (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                    elif constr == 'mix':       
                        if rev_idx <= 333:
                            m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 <= (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                        elif rev_idx <= 666:
                            m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 >= (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                    elif constr == 'mono25' and rev_idx >= 250:
                        m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 == (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                    elif constr == 'mono50' and rev_idx >= 500:
                        m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 == (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)
                    elif constr == 'mono75' and rev_idx >= 750:
                        m.addConstr((x[idx2]+eps[idx_eps2]-x[idx1]-eps[idx_eps1]) * mul1 == (x[idx3]+eps[idx_eps3]-x[idx2]-eps[idx_eps2]) * mul2)

        
        m.params.OutputFlag = 0
        m.optimize()

        if m.status == 13:
            print("program returns a suboptimal solution")
        elif m.status != 2:
            print("program cannot be optimized with status", m.status)

        x = np.asarray([m.getVarByName("x[{}]".format(i)).x for i in range(n_paper)])
        eps = np.asarray([m.getVarByName("eps[{}]".format(i)).x for i in range(eps_len)])
        top_indices = np.argsort(x)[-int(n_paper * top_percent):]

        return m.objVal, x, top_indices

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')


def LSC_mono(papers, reviewers, top_percent=0.1, constr='linear'):
    result = min_noise(papers, reviewers, False, top_percent, constr)

    if result is None:
        return 0, 0, 0, 0

    obj, x, top_indices = result
    model_score, model_acc = process_top_indices(papers, top_indices)
    ranking_scores = ranking_metric(x, top_percent)

    return model_score, model_acc, x, ranking_scores

def LSC_card(papers, reviewers, top_percent=0.1, constr='linear'):
    result = min_noise(papers, reviewers, True, top_percent, constr)

    if result is None:
        return 0, 0, 0, 0

    obj, x, top_indices = result
    model_score, model_acc = process_top_indices(papers, top_indices)
    ranking_scores = ranking_metric(x, top_percent)

    return model_score, model_acc, x, ranking_scores

def qp_linear(papers, reviewers, cardinal=False, top_percent=0.1, constr='linear'):
    n_paper = len(papers)
    n_rev = len(reviewers)
    paper_per = len(reviewers[0]["paper_indices"])
    eps_len = len(reviewers) * paper_per

    try:
        m = gp.Model("minnoise")
        x = m.addMVar(n_paper, ub=100, name="x")
        p = m.addMVar(n_rev, name="p")
        q = m.addMVar(n_rev, lb=-10)
        eps = m.addMVar(eps_len, lb=-100, name="eps")

        m.addConstr(p.sum() == n_rev)
        m.addConstr(p >= 0)
        m.setObjective(eps @ eps, GRB.MINIMIZE)

        for rev_idx, rev in enumerate(reviewers):
            indices = rev["paper_indices"]
            scores = rev["rev_scores"]

            base_idx = rev_idx * paper_per

            for i in range(len(indices)):
                idx_i, yi = indices[i], scores[i]
                idx_eps_i = base_idx + i

                m.addConstr(eps[idx_eps_i] == yi * p[rev_idx] - q[rev_idx] - x[idx_i] )
        
        m.params.OutputFlag = 0
        m.optimize()

        if m.status == 13:
            print("program returns a suboptimal solution")
        elif m.status != 2:
            print("program cannot be optimized with status", m.status)

        x = np.asarray([m.getVarByName("x[{}]".format(i)).x for i in range(n_paper)])
        eps = np.asarray([m.getVarByName("eps[{}]".format(i)).x for i in range(eps_len)])
        top_indices = np.argsort(x)[-int(n_paper * top_percent):]

        return m.objVal, x, top_indices

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')


def QP(papers, reviewers, top_percent=0.1, constr='linear'):
    result = qp_linear(papers, reviewers, True, top_percent)

    if result is None:
        return 0, 0, 0, 0

    obj, x, top_indices = result
    model_score, model_acc = process_top_indices(papers, top_indices)
    ranking_scores = ranking_metric(x, top_percent)

    return model_score, model_acc, x, ranking_scores

def bayesian(papers, reviewers, top_percent=0.1, constr=None):
    ### Bayesian model from Ge et al.
    try:
        data = []
        paper_idx = 0
        for paper in papers:
            paper_idx += 1
            truequality = paper['true_score']
            
            for reviewer_idx, score in zip(paper['reviewers'], paper['rev_scores']):  
                data.append( [str(paper_idx), str(reviewer_idx+10000), score, truequality] )
        
        reviews = pd.DataFrame(data, columns=[ "PaperID","Email","Quality","TrueQuality" ])
        
        mu = reviews.Quality.mean()
        r = reviews
        X1 = pd.get_dummies(r.PaperID)
        X1 = X1[sorted(X1.columns, key=int)]
        X2 = pd.get_dummies(r.Email)
        X2 = X2[sorted(X2.columns, key=str.lower)]
        y = reviews.Quality - mu

        X = X1.join(X2)
        kern1 = GPy.kern.Linear(input_dim=len(X1.columns), active_dims=np.arange(len(X1.columns)))
        kern1.name = 'K_f'
        kern2 = GPy.kern.Linear(input_dim=len(X2.columns), active_dims=np.arange(len(X1.columns), len(X.columns)))
        kern2.name = 'K_b'

        model = GPy.models.GPRegression(X, y[:, None], kern1+kern2)
        model.optimize()

        alpha_f = model.sum.K_f.variances
        alpha_b = model.sum.K_b.variances/alpha_f
        sigma2 = model.Gaussian_noise.variance/alpha_f

        K_f = np.dot(X1, X1.T)
        K_b = alpha_b*np.dot(X2, X2.T)
        K = K_f + K_b + sigma2*np.eye(X2.shape[0])
        Kinv, L, Li, logdet = GPy.util.linalg.pdinv(K) # since we have GPy loaded in use their positive definite inverse.
        y = reviews.Quality - mu
        alpha = np.dot(Kinv, y)
        yTKinvy = np.dot(y, alpha)
        alpha_f = yTKinvy/len(y)

        ll = 0.5*len(y)*np.log(2*np.pi*alpha_f) + 0.5*logdet + 0.5*yTKinvy/alpha_f 


        K_s = K_f + np.eye(K_f.shape[0])*sigma2
        s = pd.Series(np.dot(K_s, alpha) + mu, index=X1.index)
        covs = alpha_f*(K_s - np.dot(K_s, np.dot(Kinv, K_s)))

        number_accepts = int(top_percent * len(papers))

        score = np.random.multivariate_normal(mean=s, cov=covs, size=1000).T
        
        paper_score = pd.DataFrame(np.dot(np.diag(1./X1.sum(0)), np.dot(X1.T, score)), index=X1.columns)
        prob_accept = ((paper_score>paper_score.quantile(1-(float(number_accepts)/paper_score.shape[0]))).sum(1)/1000)
        prob_accept.name = 'AcceptProbability'

        raw_score = pd.DataFrame(np.dot(np.diag(1./X1.sum(0)), np.dot(X1.T, r.Quality)), index=X1.columns)
        true_score = pd.DataFrame(np.dot(np.diag(1./X1.sum(0)), np.dot(X1.T, r.TrueQuality)), index=X1.columns)

        s1 = prob_accept.nlargest(number_accepts)
        s2 = true_score.nlargest(number_accepts, 0)
        c1 = s1.index.intersection(s2.index)

        model_acc = len(c1)/number_accepts
        # model_score = np.sum(true_score.loc[s1.index].to_numpy())
        model_score = np.average(true_score.loc[s1.index].to_numpy())

        x = prob_accept.to_numpy()
        ranking_scores = ranking_metric(x, top_percent)
    except:
        print("Encountered an error")
        return 0, 0, 0, 0

    return model_score, model_acc, x, ranking_scores
