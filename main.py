import models
from models import *
from data_generation import *
from utils import *
import argparse
import csv
import pandas as pd
import time
import os

from collections import defaultdict 

parser = argparse.ArgumentParser(description="Paper calibration experiments")
parser.add_argument(
    "--all",
    required=False,
    action="store_true",
    help="test all parameters",
)
parser.add_argument(
    "--model-name",
    "-m",
    required=False,
    default="bayesian",
    help="the model to test",
)

np.set_printoptions(precision=3)

def test_dir(model_name, base_dir, top_percent=0.1, start_trial=0, constr='linear'):
    model = getattr(models, model_name)

    stats = {}

    # print(f"Parameter {n_paper}/{n_rev}/{paper_per}/{noise_std}")

    counter = 0
    trials = [ f.name for f in os.scandir(base_dir) if f.is_dir() ]
    for trial in trials:
        print(trial)
        papers, reviewers = load_data(f"{base_dir}/{trial}/")
        
        baseline_score, baseline_acc, baseline_ranking_scores = baseline(papers, top_percent)

        optimal_score = optimal(papers, top_percent)
        # try:
        model_score, model_acc, x, ranking_scores = model(papers, reviewers, top_percent, constr)

        # save_results(x, f"{trial}/{model_name}_x.pickle")

        print( 'baseline_score', baseline_score )
        print( 'model_score', model_score )
        print( 'optimal_score', optimal_score )
        print( 'baseline_ranking_scores', baseline_ranking_scores)
        print( 'model_ranking_scores', ranking_scores)

        stats[trial] = {
            'baseline_score': baseline_score,
            'model_score': model_score,
            'baseline_acc': baseline_acc,
            'model_acc': model_acc,
            'optimal_score': optimal_score,
            'baseline_map': baseline_ranking_scores[0],
            'baseline_ndcg': baseline_ranking_scores[1],
            'baseline_l1': baseline_ranking_scores[2],
            'model_map': ranking_scores[0],
            'model_ndcg': ranking_scores[1],
            'model_l1': ranking_scores[2],
        }
  

    return stats


def test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1, start_trial=0, constr='linear'):
    model = getattr(models, model_name)

    stats = defaultdict(list)

    print(f"Parameter {n_paper}/{n_rev}/{paper_per}/{noise_std}")

    counter = 0
    for trial in range(start_trial, 20):
        if trial % 10 == 9:
            print(trial)

        dir = "./data/{}/{}_{}_{}_{}_{}/trial{}/".format(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, trial)
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist")
        papers, reviewers = load_data(dir)

        # G, connected = construct_graph(reviewers)
        # max_pun = check_maximum_set(reviewers)
        # print(connected, max_pun)
        
        baseline_score, baseline_acc, _ = baseline(papers, top_percent)

        optimal_score = optimal(papers, top_percent)
        try:
            model_score, model_acc, x, ranking_scores = model(papers, reviewers, top_percent, constr)

            save_results(x, f"{dir}{model_name}_x.pickle")

            if model_score > 0:
                stats['baseline_score'].append(baseline_score)
                stats['model_score'].append(model_score)
                stats['optimal_score'].append(optimal_score)
                stats['model_map_scores'].append(ranking_scores[0])
                stats['model_ndcg_scores'].append(ranking_scores[1])
                stats['model_l1_scores'].append(ranking_scores[2])

                stats['baseline_acc'].append(baseline_acc)
                stats['model_acc'].append(model_acc)

                stats['diff_bls'].append(optimal_score - baseline_score)
                stats['diff_mds'].append(optimal_score - model_score)

                # print(baseline_acc, model_acc, ranking_scores)
                counter += 1
                if counter == 20:
                    break
            else:
                print(f"Trial {trial} encounters an error")

        except Exception as inst:
            print("============================")
            print(inst)

    toc = time.perf_counter()

    print(f"Elapsed time: {toc - tic}")

    for key, value in stats.items():
        print(f"{key}: {np.mean(value)}, {np.std(value)}")

    stats['model_std'] = np.std(stats['model_acc'])
    stats['baseline_std'] = np.std(stats['baseline_acc'])
    stats['model_score_std'] = np.std(stats['model_score'])
    stats['baseline_score_std'] = np.std(stats['baseline_score'])
    stats['model_rel_std'] = np.std(stats['model_rel'])
    stats['baseline_rel_std'] = np.std(stats['baseline_rel'])
    stats['optimal_score_std'] = np.std(stats['optimal_score'])

    stats['model_map_std'] = np.std(stats['model_map_scores'])
    stats['model_ndcg_std'] = np.std(stats['model_ndcg_scores'])
    stats['model_l1_std'] = np.std(stats['model_l1_scores'])

    stats['model_acc'] = np.mean(stats['model_acc'])
    stats['baseline_acc'] = np.mean(stats['baseline_acc'])
    stats['model_score'] = np.mean(stats['model_score'])
    stats['baseline_score'] = np.mean(stats['baseline_score'])
    stats['model_rel'] = np.mean(stats['model_rel'])
    stats['baseline_rel'] = np.mean(stats['baseline_rel'])

    stats['model_map_scores'] = np.mean(stats['model_map_scores'])
    stats['model_ndcg_scores'] = np.mean(stats['model_ndcg_scores'])
    stats['model_l1_scores'] = np.mean(stats['model_l1_scores'])

    fname = "./data/{}/{}_{}_{}_{}_{}/{}.json".format(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, model_name)
    with open(fname, 'w') as outfile:
        json.dump(stats, outfile, indent=4)   

    return stats

def test_paper_per(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'LSC_mono': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'LSC_card': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'QP': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        #'bayesian': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
    }

    for model_name in ['LSC_mono', 'LSC_card', 'QP']:
        for paper_per in [3,4,5,6,7]:
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])
    
    fname = 'paper_per.json' if noise_std == 0 else 'paper_per_noisy.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)

def test_rev_size(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'LSC_mono': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'LSC_card': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'QP': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        #'bayesian': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
    }
    for model_name in ['LSC_mono', 'LSC_card', "QP"]:
        for n_rev in [500, 750, 1000]:
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])

    fname = 'rev_size.json' if noise_std == 0 else 'rev_size_noisy.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)

def test_noise(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'LSC_mono': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'LSC_card': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'QP': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        # 'bayesian': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
    }
    for model_name in ['LSC_mono', 'LSC_card', 'QP']:
        for noise_std in [0, 0.25, 0.5, 0.75, 1]:
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])

    with open('noise.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)


def test_paper_per_bay(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'bayesian': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
    }

    for model_name in ['bayesian']:
        for paper_per in [3,4,5,6,7]:
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])
    
    fname = 'paper_per_bay.json' if noise_std == 0 else 'paper_per_bay_noisy.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)


def test_rev_size_bay(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'bayesian': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
    }
    for model_name in ['bayesian']:
        for n_rev in [500, 750, 1000]:
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])

    fname = 'rev_size_bay.json' if noise_std == 0 else 'rev_size_bay_noisy.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)


def test_noise_bay(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'bayesian': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
    }
    for model_name in ['bayesian']:
        for noise_std in [0, 0.25, 0.5, 0.75, 1]:
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])

    with open('noise_bay.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)


def test_conf_size(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'model1_ord': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'model1_card': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []}
    }
    for model_name in ['LSC_mono', 'LSC_card', "QP"]:
        for n_paper in [100, 1000, 5000]:
            n_rev = n_paper
            stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])

    fname = 'conf_size.json' if noise_std == 0 else 'conf_size_noisy.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)


def test_linear_mono_mix(n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'LSC_mono': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'LSC_linear': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'LSC_mix': {'baseline_acc': [], 'model_acc': [], 'baseline_std' : [], 'model_std': []},
        'QP' : {'baseline_acc' : [], 'model_acc' : [], 'baseline_std' : [], 'model_std' : []}
    }

    for base_dir in ['mono100', 'mono75', 'mono50', 'mono25']:
        print(f"Testing on dir {base_dir}")
        for model_name in ['LSC_linear', 'LSC_mono', 'LSC_mix', 'QP']:
            print(f"Testing on model {model_name}")
            if model_name == 'LSC_linear':
                stats = test_trials('LSC_card', base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1, constr='linear')
            elif model_name == 'LSC_mix':
                if base_dir == "mono100":
                    stats = test_trials('LSC_mono', base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1)
                else:
                    stats = test_trials('LSC_card', base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1, constr=base_dir)
            else:
                stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])
            result[model_name]['model_std'].append(stats['model_std'])
            result[model_name]['stats'] = stats

    fname = 'mix.json' if noise_std == 0 else 'mix_noise.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)


def test_linear_mono_mix_bay(n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1):
    result = {
        'bayesian' : {'baseline_acc' : [], 'model_acc' : [], 'baseline_std' : [], 'model_std' : []}
    }

    for base_dir in ['mono100', 'mono75', 'mono50', 'mono25']:
        print(f"Testing on dir {base_dir}")
        for model_name in ['bayesian']: #['LSC_linear', 'LSC_mono', 'LSC_mix', 'QP']:
            print(f"Testing on model {model_name}")
            if model_name == 'LSC_linear':
                stats = test_trials('LSC_card', base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1, constr='linear')
            elif model_name == 'LSC_mix':
                if base_dir == "mono100":
                    stats = test_trials('LSC_mono', base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1)
                else:
                    stats = test_trials('LSC_card', base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1, constr=base_dir)
            else:
                stats = test_trials(model_name, base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, top_percent=0.1) 
            result[model_name]['baseline_acc'].append(stats['baseline_acc'])
            result[model_name]['model_acc'].append(stats['model_acc'])
            result[model_name]['baseline_std'].append(stats['baseline_std'])

    fname = 'mix_bay.json' if noise_std == 0 else 'mix_bay_noise.json'
    with open(fname, 'w') as outfile:
        json.dump(result, outfile, indent=4)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

if __name__ == '__main__':
    args = parser.parse_args()
    model = getattr(models, args.model_name)

    # playground mode 
    tic = time.perf_counter()

    n_paper = 1000
    n_rev = 1000

    paper_std = 1.2
    noise_std = 0

    top_percent = 0.1


    stats = defaultdict(list)

    for n_paper in [1000]:
        counter = 0
        print(f"Parameter {n_paper}/{n_rev}/{paper_per}/{noise_std}")
        for trial in range(20):
            if trial % 10 == 9:
                print(trial)

            dir = "./data/{}/{}_{}_{}_{}_{}/trial{}/".format(base_dir, n_paper, n_rev, paper_per, paper_std, noise_std, trial)
            if not os.path.exists(dir):
                print(f"Directory {dir} does not exist")
            papers, reviewers = load_data(dir)
            
            baseline_score, baseline_acc, baseline_ranking = baseline(papers, top_percent)

            optimal_score = optimal(papers, top_percent)
            model_score, model_acc, x, ranking_scores = model(papers, reviewers, top_percent, constr)

            if model_score > 0:
                stats['baseline_acc'].append(baseline_acc)
                stats['model_acc'].append(model_acc)

                stats['baseline_rel'].append(baseline_score / optimal_score)
                stats['model_rel'].append(model_score / optimal_score)

                stats['diff_bls'].append(optimal_score - baseline_score)
                stats['diff_mds'].append(optimal_score - model_score)

                stats['baseline_map_scores'].append(baseline_ranking[0])
                stats['baseline_ndcg_scores'].append(baseline_ranking[1])
                stats['baseline_l1_scores'].append(baseline_ranking[2])

                stats['model_map_scores'].append(ranking_scores[0])
                stats['model_ndcg_scores'].append(ranking_scores[1])
                stats['model_l1_scores'].append(ranking_scores[2])

                print(baseline_acc, model_acc)
                counter += 1
                if counter == 20:
                    break
            else:
                print(f"Trial {trial} encounters an error")

        toc = time.perf_counter()

        print(f"Elapsed time: {toc - tic}")

        for key, value in stats.items():
            print(f"{key}: {np.mean(value)}, {np.std(value)}")

        stats = defaultdict(list)
