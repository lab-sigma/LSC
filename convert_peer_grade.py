import csv, json, os


max_grades = {}
with open('peer-grading-dataset/maxgrades.csv', 'r') as f:
    rows = csv.reader(f, delimiter=',')
    next(rows)
    for row in rows:
    	taskid, maxgrade = row   
    	max_grades[taskid] = int(maxgrade)

tasks = {}

def normalize(grade, max_grades): ### to [-10, 10]
	return (grade/max_grades - 0.5)*10

# mode = 0 # by question
mode = 1 # by homework
# mode = 2 # by semester

def hash(taskid, groupid, mode=0):
	if mode == 0:
		return taskid, groupid
	elif mode == 1:
		return taskid[0], f"{groupid}_{taskid[1]}" 
	elif mode == 2:
		return 1, f"{groupid}_{taskid}" 

	return None

with open('peer-grading-dataset/peer.csv', 'r') as f:
    rows = csv.reader(f, delimiter=',')
    next(rows)
    for row in rows:

    	taskid,reviewerid,groupid,grade = row
    	grade = normalize( int(grade), max_grades[taskid])

    	taskid, groupid = hash(taskid, groupid, mode=mode)

    	if taskid not in tasks:
    		tasks[taskid] = {'papers': {}, 'reviewers': {}}

    	if groupid not in tasks[taskid]['papers']:
    		tasks[taskid]['papers'][groupid] = []

    	tasks[taskid]['papers'][groupid].append( (reviewerid, grade) )

    	if reviewerid not in tasks[taskid]['reviewers']:
    		tasks[taskid]['reviewers'][reviewerid] = []

    	tasks[taskid]['reviewers'][reviewerid].append([groupid, grade])

ta_tasks = {}
with open('peer-grading-dataset/ta.csv', 'r') as f:
    rows = csv.reader(f, delimiter=',')
    next(rows)
    for row in rows:
    	taskid,reviewerid,groupid,grade = row
    	grade = normalize( int(grade), max_grades[taskid])

    	taskid, groupid = hash(taskid, groupid, mode=mode)

    	if taskid not in ta_tasks:
    		ta_tasks[taskid] = {}

    	if groupid not in ta_tasks[taskid]:
    		ta_tasks[taskid][groupid] = {}

    	ta_tasks[taskid][groupid][reviewerid] = grade



for taskid, results in tasks.items():
	path = f'data/peer-grading/{taskid}/'
	if not os.path.exists(path):
		os.mkdir(path) 

	reviewer_id_map = {}
	papar_id_map = {}

	paper_scores = [] 		### use TA's average  by default
	for paperid, reviews in ta_tasks[taskid].items():
		avg_score = 0
		count = 0
		for ta, score in reviews.items():
			avg_score += score
			count += 1
		avg_score /= count
		paper_scores.append( (paperid, avg_score) )

	paper_scores.sort(key=lambda tup: tup[1])
	idx = 0
	for p in paper_scores:
		papar_id_map[p[0]] = idx ### normalized, ranked id
		idx += 1
	
	num_papers = len(paper_scores)
	papers = [None]*num_papers

	idx = 0 ### normalized reviewer id
	for paperid, reviews in results['papers'].items():
		# if paperid not in papar_id_map:
		# 	print(paperid)
		# 	continue

		paper =  {
        "reviewers": [],
        "rev_scores": [],
        "true_score": paper_scores[ papar_id_map[paperid] ][1] 
        }
		for reviewerid, grade  in reviews:
			if reviewerid not in reviewer_id_map:
				reviewer_id_map[reviewerid] = idx
				idx += 1
			paper['reviewers'].append( reviewer_id_map[reviewerid] )
			paper['rev_scores'].append(grade)

		papers[ papar_id_map[paperid] ] = paper

	with open(f'data/peer-grading/{taskid}/papers.json', 'w') as f:
		json.dump(papers, f, indent=4)

	num_reviewers = len(reviewer_id_map.keys())
	reviewers = [None]*num_reviewers

	for reviewerid, papers in results['reviewers'].items():
		reviewer =  {
		"paper_indices": [],
		"rev_scores": [],
		"baseline": None
		}
		papers = sorted( [(paperid, grade) for paperid, grade  in papers ], key=lambda tup: tup[1] )
		for paperid, grade  in papers:
			reviewer['paper_indices'].append( papar_id_map[paperid] )
			reviewer['rev_scores'].append(grade)
		
		reviewers[ reviewer_id_map[reviewerid] ] = reviewer

	with open(f'data/peer-grading/{taskid}/reviewers.json', 'w') as f:
		json.dump(reviewers, f, indent=4)

	print(taskid, num_papers, num_reviewers)