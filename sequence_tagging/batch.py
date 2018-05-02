import subprocess
import shlex

with open("single.out", 'a') as f:
	for iteration in range(1,14, 2):
		for step in range(1,3):
			subprocess.call(shlex.split("nice -n 19 nohup python train.py "+str(iteration)+" "+str(step)), stdout=f, stderr=f)
			subprocess.call(shlex.split("nice -n 19 nohup python evaluate.py "+str(iteration)+" "+str(step)), stdout=f, stderr=f)