import numpy as np
import matplotlib.pyplot as plt
import re
import fnmatch
import os

def read_file(filename, pattern="eval_avg_score"):
	table = []
	with open(filename) as f:
		text = f.read()
		text = re.split('\n', text)
		for row in text:
			if pattern in row:
				table.append(re.split(' +', row))

	data = []
	for row in table:
		if row[-2] != "":
			data.append(np.float32(row[-2]))
	nRows = len(data)
	if "avg_score" in pattern:
		if 'WIP' in filename:
			time = np.asarray([i*2 for i in range(nRows)])
		else:
			time = np.asarray([i*.2 for i in range(nRows)])
	else:
		if 'WIP' in filename:
			time = np.asarray([i*16.0/1000 for i in range(nRows)])
		else:
			time = np.asarray([i*4.0/1000 for i in range(nRows)])
	return time, data

# log_files = sorted(fnmatch.filter(os.listdir('.'), "*log*.txt"))
log_files = ["2017-11-02-17-31-06-331713-log.txt",
			 "2017-11-02-19-55-40-032464-log.txt",
			 "2017-11-05-11-57-30-812910-log-WIP.txt",
			 "2017-11-05-12-05-20-545401-log-WIP.txt"]
labels = ['Step 1. Switch to Q function',
		  'Step 2. Switch ADV from (Yn-V) to (Q-V)',
		  'Step 2. (Q-V), parallel by 32',
		  'Step 2. (Q-V), parallel by 32']
def visualize(log_files, labels, metric):
	for filename, label in zip(log_files, labels):
		time, data = read_file(filename, metric)
		if "explained" in metric:
			data = np.maximum(data, -2)
		plt.plot(time, data, label=label)
		if "explained" in metric:
			plt.ylim([-2,1])
	plt.legend()
	plt.show()
	plt.close()

for metric in ["eval_avg_score", "explained_variance", "value_loss", "entropy"]:
	visualize(log_files, labels, metric)
