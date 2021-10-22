import numpy as np
import pickle, math


def baseline(papers, top_percent=0.1):
    n_paper = len(papers)
    true_scores = np.asarray([p["true_score"] for p in papers])
    baselines = []
    for i in range(len(papers)):
        baseline = np.mean(papers[i]['rev_scores']) if len(papers[i]['rev_scores']) > 0 else 0.0
        papers[i]['baseline'] = baseline
        baselines.append(baseline)

    topk = np.argsort(baselines)[-int(top_percent * n_paper):]
    sum = 0
    for idx in topk:
        if (1-top_percent)*len(papers) <= idx <= len(papers):
            sum += 1
    acc = sum / (len(papers) * top_percent)
    
    topk_scores = [true_scores[idx] for idx in topk]
    return np.average(topk_scores), acc, ranking_metric(np.asarray(baselines) )


def optimal(papers, top_percent=0.1):
    length = int(len(papers) * top_percent)
    true_scores = np.asarray([p["true_score"] for p in papers])
    best_score = np.average(true_scores[-length:])

    return best_score


def process_top_indices(papers, top_indices):
    true_scores = np.asarray([p["true_score"] for p in papers])
    top_indices = np.asarray(top_indices)
    
    sum = 0
    for index in top_indices:
        if len(papers) - len(top_indices) <= index <= len(papers):
            sum += 1

    acc = sum / len(top_indices)
    # lp_score = np.sum(true_scores[top_indices])
    lp_score = np.average(true_scores[top_indices])
    
    return lp_score, acc


def manhatten(x):
    value = np.array(list(zip(x, np.arange(len(x)))), dtype=[('value', 'f4'), ('index', 'i4')])
    rank = np.argsort(value, order=['value', 'index'] )

    total = 0
    for i, _ in enumerate(x):
        total += abs(rank[i] - i)
    
    return total

def ranking_metric(x, top_percent=0.1):
    ### higher score paper has lower rank
    value = np.array(list(zip(-x, np.arange(len(x)))), dtype=[('value', 'f4'), ('index', 'i4')])
    rank = np.argsort(value, order=['value', 'index'] )
    
    l1 = 0
    for i, _ in enumerate(x):
        l1 += abs(rank[i] - (len(x)-1-i) )
    
    thresholds = int(len(x)*top_percent)
    num_relevant = 0
    dcg_score = 0
    map_score = 0
    idcg_score = 0
    
    for i in range(thresholds):
        idcg_score += 1 / math.log(i+2) 
        if rank[i] >= len(x)-thresholds: ###  the relevant item is retrieved 
            num_relevant += 1
            map_score += num_relevant /(i+1)
            
            relevance_rating = 1 ## could change use rank[i] to mimic the rating             
            dcg_score += (2**relevance_rating - 1) / math.log(i+2) 
    
    if num_relevant != 0:
        map_score /= num_relevant

    ndcg_score = dcg_score / idcg_score
    
    return map_score, ndcg_score, l1
    
    
def save_results(x, file):
    with open(file, 'wb') as f:
        pickle.dump(x, f)