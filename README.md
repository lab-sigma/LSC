# Least Square Calibration for Peer Reviews

Requirements
- gurobipy       -> for solving convex programs
- GPy            -> for Bayesian baseline
- numpy
- pandas


To generate paper review data, execute
```
python generate_data.py
```
which will generate 20 trials of paper review data in the `data/linear` folder. 

To generate data from the peer grading dataset, execute
```
python convert_peer_grade.py
```

To run a model on these 20 trials, execute
```
python main.py --m $model
```
where $model can be one of `LSC_mono, LSC_card, QP, bayesian`

To run a model in the noisy setting, change `noise_std` at line 183 of `main.py` to 0.5, and then execute the command above.

