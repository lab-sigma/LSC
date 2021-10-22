import random
import numpy as np
import json, os

"""
Data Module: randomly generate paper and reviewer data

Here are the data structures that store paper and reviewer information:

papers = [
    {
        "true_score" : float,
        "reviewers" : [],
        "rev_scores" : [],
    },

        "true_score" : float,
        "reviewers" : [],
        "rev_scores" : [],
    },
    ...
]

reviewers = [
    {
        "baseline" : float (baseline score for reviewer. Reviewer function is concave for papers above the baseline score and convex otherwise )
        "paper_indices" : [],
        "rev_scores" : [],
        "func_pos" : function,
        "func_neg" : function,
        "rev_scores_pos" : [(idx, rev_score),], 
        "rev_scores_neg : [(idx, rev_score),],
    },
    ...
]
"""

def generate_assignment(n_paper, n_rev, paper_per):
    assignment = []
    assignment.append( [1,2,3] )
    assignment.append( [1,2,4] )
    
    usage = [paper_per] ** n_paper
    usage[1] -= 2
    usage[2] -= 2
    usage[3] -= 1
    usage[4] -= 1
    for i in range(3, n_rev):
        assignment.append( [i, i+2, n_rev+i//2, 1.5*n_rev+i, 2.5*rev+i ] )


def generate_component(n_paper, n_rev, paper_per, conectivity=2):
    ### form a doubly connect component and grow it
    assignments = []
    assigned = set()  
    done = set()  
    unassigned = set(range(n_paper))
    usage = {}
    bar = n_rev * paper_per // n_paper

    usage = np.zeros(n_paper, dtype=int)
    papers = list(sorted(np.random.choice(n_paper, paper_per, replace=False)))
    assignments.append(papers)
    for p in papers: 
        assigned.add(p)
        unassigned.remove(p)  
        usage[p] += 1
        if usage[p] >= bar:
            done.add(p)


    for i in range(1, n_rev):
        p2 = None
        if len(unassigned) >= paper_per - conectivity:
            p2 = np.random.choice(list(unassigned), paper_per-conectivity, replace=False)
        else:
            p2 = np.asarray(list(unassigned), dtype=int)

        assigned_notdone = list(assigned.difference(done))
        assigned_done = list(assigned.intersection(done) )
        p1 = None
        if len(assigned_notdone) >= conectivity:
            p1 = np.random.choice(assigned_notdone, conectivity, replace=False)
        else:
            p1 = np.asarray(assigned_notdone, dtype=int)

        if len(p1)+len(p2) < paper_per:
            p0 = np.random.choice(assigned_done, paper_per-len(p1)-len(p2), replace=False)
            p1 = np.concatenate([p1, p0], dtype=int )

        for p in p2: 
            assigned.add(p)
            unassigned.remove(p)

        papers = np.concatenate([p1, p2], dtype=int ) if p2.shape[0] > 0 else p1
        for p in papers: 
            usage[p] += 1
            if usage[p] >= bar:
                done.add(p)

        # print(papers)
        assignments.append( list(sorted(papers)) )


    return assignments


def generate_triply(n_paper, n_rev):
    assert n_paper == 1000
    assert n_rev == 499

    assignments = []
    idx = 0
    for i in range(498):
        assignments.append([idx, idx+1, idx+2, idx+3, idx+4])
        idx += 2
    assignments.append([996,997,998,999])

    return assignments


def generate_doubly(n_paper, n_rev):
    assert n_paper == 1000
    assert n_rev == 499

    assignments = []
    idx = 0
    for i in range(332):
        assignments.append([idx, idx+1, idx+2, idx+3, idx+4])
        idx += 3
    assignments.append([996,997,998,999])
    
    for i in range(333, 499):
        papers = list(sorted(np.random.choice(1000, 5, replace=False)))
        assignments.append(papers)

    return assignments


def generate_doubly_triply(n_paper, n_rev, paper_per, paper_std=1, noise_std=0.5, true_scores=None, rev_funcs_param=None):
    papers_doubly = [{"reviewers" : [], "rev_scores" : []} for _ in range(n_paper)]
    reviewers_doubly = [{"paper_indices" : [], "rev_scores" : []} for _ in range(n_rev)]
    papers_triply = [{"reviewers" : [], "rev_scores" : []} for _ in range(n_paper)]
    reviewers_triply = [{"paper_indices" : [], "rev_scores" : []} for _ in range(n_rev)]

    true_scores = np.random.normal(loc=5.32, scale=paper_std, size=(n_paper))
    true_scores[true_scores < 0] = 0
    true_scores[true_scores > 10] = 10

    true_scores = sorted(true_scores)

    for i in range(n_paper):
        papers_doubly[i]["true_score"] = true_scores[i]
        papers_triply[i]["true_score"] = true_scores[i]

    # doubly_assignments = generate_doubly(n_paper, n_rev)
    # triply_assignments = generate_triply(n_paper, n_rev)
    doubly_assignments = generate_component(n_paper, n_rev, paper_per, conectivity=2)
    triply_assignments = generate_component(n_paper, n_rev, paper_per, conectivity=3)

    rev_funcs = [generate_reviewer_func_linear() for i in range(n_rev)]

    for i in range(n_rev):
        baseline = 5.32
        reviewers_doubly[i]["baseline"] = baseline
        
        rev_func = rev_funcs[i]
        rev_scores = []
        for j, idx in enumerate(doubly_assignments[i]):
            # print(j,idx)
            true_score = papers_doubly[idx]["true_score"]
            eps = np.random.normal(loc=0, scale=noise_std)
            
            true_score += eps
            true_score = min(10, max(0, true_score))
            rev_score = rev_func(true_score)

            papers_doubly[idx]["rev_scores"].append(rev_score)
            rev_scores.append(rev_score)
            papers_doubly[idx]["reviewers"].append(i)

        sorted_pairs = sorted(zip(doubly_assignments[i], rev_scores), key=lambda x : x[1])
        paper_indices = [sorted_pairs[i][0] for i in range(len(sorted_pairs))]
        rev_scores = [sorted_pairs[i][1] for i in range(len(sorted_pairs))]

        reviewers_doubly[i]["paper_indices"] = paper_indices
        reviewers_doubly[i]["rev_scores"] = rev_scores

    for i in range(n_rev):
        baseline = 5.32
        reviewers_triply[i]["baseline"] = baseline

        rev_func = rev_funcs[i]
        rev_scores = []
        for j, idx in enumerate(triply_assignments[i]):
            true_score = papers_triply[idx]["true_score"]
            eps = np.random.normal(loc=0, scale=noise_std)
            
            true_score += eps
            true_score = min(10, max(0, true_score))
            rev_score = rev_func(true_score)

            papers_triply[idx]["rev_scores"].append(rev_score)
            rev_scores.append(rev_score)
            papers_triply[idx]["reviewers"].append(i)

        sorted_pairs = sorted(zip(triply_assignments[i], rev_scores), key=lambda x : x[1])
        paper_indices = [sorted_pairs[i][0] for i in range(len(sorted_pairs))]
        rev_scores = [sorted_pairs[i][1] for i in range(len(sorted_pairs))]

        reviewers_triply[i]["paper_indices"] = paper_indices
        reviewers_triply[i]["rev_scores"] = rev_scores


    return papers_doubly, reviewers_doubly, papers_triply, reviewers_triply



def generate_data(n_paper, n_rev, paper_per, paper_std=1, noise_std=0.5, true_scores=None, rev_funcs_param=None):

    papers = [{"reviewers" : [], "rev_scores" : []} for _ in range(n_paper)]
    reviewers = [{"paper_indices" : [], "rev_scores" : []} for _ in range(n_rev)]

    if true_scores is None:
        true_scores = np.random.normal(loc=5.32, scale=paper_std, size=(n_paper))
        #true_scores = np.random.uniform(low=0, high=10, size=(n_paper))
        true_scores[true_scores < 0] = 0
        true_scores[true_scores > 10] = 10
    
    true_scores = sorted(true_scores)

    for i in range(n_paper):
        papers[i]["true_score"] = true_scores[i]

    paper_pool = set(np.arange(n_paper))
    paper_counter = [0 for i in range(n_paper)]

    for i in range(n_rev):
        baseline = 5.32
        # baseline = baselines[i]

        reviewers[i]["baseline"] = baseline

        if len(paper_pool) < paper_per:
            assignment = list(paper_pool) + list(np.random.choice(n_paper, paper_per - len(paper_pool), replace=False))
        else:
            assignment = np.random.choice(tuple(paper_pool), paper_per, replace=False)
        
        for idx in assignment:
            paper_counter[idx] += 1
            if paper_counter[idx] == paper_per*n_rev//n_paper: ## paper_per
                paper_pool.remove(idx)
        
        # if i <= 250:
        #     rev_func_pos = generate_reviewer_func_concave()
        # elif i <= 500:
        #     rev_func_pos = generate_reviewer_func_convex()
        # else:
        #     rev_func_pos = generate_monotone_func()
        rev_func = generate_reviewer_func_linear()

        rev_scores = []

        assignment = sorted(assignment)
        for j, idx in enumerate(assignment):
            true_score = papers[idx]["true_score"]
            eps = np.random.normal(loc=0, scale=noise_std)
            
            # true_score += eps
            # true_score = min(10, max(-10, true_score))

            # if i > 500:
            #     rev_score = rev_func(j)
            rev_score = rev_func(true_score)
            rev_score += eps
            #reviewers[i]["rev_scores_pairs"].append((idx, rev_score))

            papers[idx]["rev_scores"].append(rev_score)
            rev_scores.append(rev_score)
            papers[idx]["reviewers"].append(i)

        #reviewers[i]["rev_scores"] = sorted(reviewers[i]["rev_scores"], key=lambda x : x[1])

        sorted_pairs = sorted(zip(assignment, rev_scores), key=lambda x : x[1])
        paper_indices = [sorted_pairs[i][0] for i in range(len(sorted_pairs))]
        rev_scores = [sorted_pairs[i][1] for i in range(len(sorted_pairs))]

        reviewers[i]["paper_indices"] = paper_indices
        reviewers[i]["rev_scores"] = rev_scores

    return papers, reviewers



def generate_reviewer_func_linear(param=None):
    p = np.random.uniform(0, 2)
    b = np.random.normal(loc=0, scale=2, size=1)[0]

    def func(x):
        x = max(x, 0)
        return p * x + b

    return func


def generate_reviewer_func_convex(param=None):
    if param is None:
        c1 = random.uniform(0, 1)
        c2 = random.uniform(0, 1)
        c3 = random.uniform(0, 1)

        a = random.uniform(0, 1)
        b = random.uniform(a, 1)
        c = 1 - b
        b -= a
    else:
        a, b, c, c1, c2, c3 = param

    def func(x):
        x = max(x, 0)

        y = a * c1 * x ** (2) + b * c2 * x ** (2.5) + c * c3 * x ** (3)

        return y

    return func


def generate_reviewer_func_concave(param=None):
    if param is None:
        p1 = random.uniform(1, 10/(10**0.5))
        p2 = random.uniform(1, 10/(10**(1/3)))
        p3 = random.uniform(1, 10/(10**(1/4)))

        a = random.uniform(0, 1)
        b = random.uniform(a, 1)
        c = 1 - b
        b -= a

    def func(x):
        x = max(x, 0)

        y = a * p1 * (x ** (1/2)) + b * p2 * (x**(1/3)) + c * p3 * (x**(1/4))

        return y
    
    return func


def generate_monotone_func(param=None):
    scores = sorted(np.random.uniform(0, 10, size=5))

    def func(idx):
        return scores[idx]

    return func
        


papers_file_name = 'papers.json'
reviewers_file_name = 'reviewers.json'

def save_data(papers, reviewers, dir):
    with open(dir + papers_file_name, 'w') as outfile:
        json.dump(papers, outfile, indent=4, cls=NpEncoder)
    with open(dir + reviewers_file_name, 'w') as outfile:
        json.dump(reviewers, outfile, indent=4, cls=NpEncoder)


def load_data(dir):
    with open(dir + papers_file_name, 'r') as outfile:
        papers = json.load(outfile)
    with open(dir + reviewers_file_name, 'r') as outfile:
        reviewers = json.load(outfile)
    return papers, reviewers

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    n_trial = 20
    paper_std = 1.2

    for n_paper in [1000]:
        for n_rev in [400]:
            # n_rev = n_paper
            for paper_per in [4]:
                for noise_std in [0, 0.25, 0.5]:

                    for trial in range(n_trial):
                        true_scores = None
                        # true_scores = np.load(f"./true_scores/{n_paper}_{trial}.npy")
                        rev_funcs = None

                        papers_doubly, reviewers_doubly, papers_triply, reviewers_triply = generate_doubly_triply(n_paper, n_rev, paper_per, paper_std, noise_std, true_scores, rev_funcs)

                        dir_doubly = "./data/doubly/{}_{}_{}_{}_{}/trial{}/".format(n_paper, n_rev, paper_per, paper_std, noise_std, trial)
                        dir_triply = "./data/triply/{}_{}_{}_{}_{}/trial{}/".format(n_paper, n_rev, paper_per, paper_std, noise_std, trial)

                        if not os.path.exists(dir_doubly):
                            os.makedirs(dir_doubly)
                        if not os.path.exists(dir_triply):
                            os.makedirs(dir_triply)
                        save_data(papers_doubly, reviewers_doubly, dir_doubly)
                        save_data(papers_triply, reviewers_triply, dir_triply)


                        dir = "./data/linear3/{}_{}_{}_{}_{}/trial{}/".format(n_paper, n_rev, paper_per, paper_std, noise_std, trial)
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                            print("created directory", dir)
                            papers, reviewers = generate_data(n_paper, n_rev, paper_per, paper_std, noise_std, true_scores, rev_funcs)
                            print("generated data")
                            save_data(papers, reviewers, dir)
                            print("saved data")
                        else:
                            print("existing directory", dir)
                            continue