import pickle
tasks = ['BeamRider-v0',
		 'Breakout-v0',
		 'Enduro-v0',
		 # 'Pong-v0',
		 'Qbert-v0',
		 'Seaquest-v0',
		 'SpaceInvaders-v0']

lr = '0.01'
for task in tasks:
	file_name = 'SGD_results_' + task + '_lr_' + lr + '.pkl'
	with open(file_name,'rb') as f:
		A = pickle.load(f)
		max_score = max(A[1])
		time_totall = sum(A[3])
		hour = int(time_totall/3600.0)
		minute = int( time_totall - 3600 * hour)/60.0
		print('-'*60)
		print('max score for ', task[:-3], ' = ',max_score)
		print('time = ', hour, 'hours and ', minute, 'minutes')

lr = '1e-05'
for task in tasks:
	file_name = 'SGD_results_' + task + '_lr_' + lr + '.pkl'
	with open(file_name,'rb') as f:
		A = pickle.load(f)
		max_score = max(A[1])
		time_totall = sum(A[3])
		hour = int(time_totall/3600.0)
		minute = int( time_totall - 3600 * hour)/60.0
		print('-'*60)
		print('max score for ', task[:-3], ' = ',max_score)
		print('time = ', hour, 'hours and ', minute, 'minutes')


import pickle
file_name = 'SGD_results_Breakout-v0_lr_1000.0.pkl'
with open(file_name,'rb') as f:
	A = pickle.load(f)
	max_score = max(A[1])
	time_totall = sum(A[3])
	hour = int(time_totall/3600.0)
	minute = int( time_totall - 3600 * hour)/60.0
	print('-'*60)
	print('max score for Breakout',max_score)
	print('time = ', hour, 'hours and ', minute, 'minutes')


import pickle
file_name = 'SGD_results_Enduro-v0_lr_0.01.pkl'
with open(file_name,'rb') as f:
	A = pickle.load(f)
	max_score = max(A[1])
	time_totall = sum(A[3])
	hour = int(time_totall/3600.0)
	minute = int( time_totall - 3600 * hour)/60.0
	print('-'*60)
	print('max score for Enduro = ',max_score)
	print('time = ', hour, 'hours and ', minute, 'minutes')


import pickle
file_name = 'SGD_results_Qbert-v0_lr_0.01.pkl'
with open(file_name,'rb') as f:
	A = pickle.load(f)
	max_score = max(A[1])
	time_totall = sum(A[3])
	hour = int(time_totall/3600.0)
	minute = int( time_totall - 3600 * hour)/60.0
	print('-'*60)
	print('max score for Qbert = ',max_score)
	print('time = ', hour, 'hours and ', minute, 'minutes')


import pickle
file_name = 'SGD_results_Seaquest-v0_lr_0.01.pkl'
with open(file_name,'rb') as f:
	A = pickle.load(f)
	max_score = max(A[1])
	time_totall = sum(A[3])
	hour = int(time_totall/3600.0)
	minute = int( time_totall - 3600 * hour)/60.0
	print('-'*60)
	print('max score for Seaquest = ',max_score)
	print('time = ', hour, 'hours and ', minute, 'minutes')


import pickle
file_name = 'SGD_results_SpaceInvaders-v0_lr_0.01.pkl'
with open(file_name,'rb') as f:
	A = pickle.load(f)
	max_score = max(A[1])
	time_totall = sum(A[3])
	hour = int(time_totall/3600.0)
	minute = int( time_totall - 3600 * hour)/60.0
	print('-'*60)
	print('max score for SpaceInvaders = ',max_score)
	print('time = ', hour, 'hours and ', minute, 'minutes')







import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size': 16})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)
rc('axes', labelsize=30)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tasks = ['BeamRider-v0',
		 'Breakout-v0',
		 'Enduro-v0',
		 # 'Pong-v0',
		 'Qbert-v0',
		 'Seaquest-v0',
		 'SpaceInvaders-v0']

mean_scores = []
std_scores = []

for task in tasks:
	mean_scores.append(np.mean(max_score_simulations[task],axis=0))
	std_scores.append(np.std(max_score_simulations[task],axis=0))

cv_score = [0]*6
for i in range(len(std_time)):
	cv_score[i] = std_scores[i] / mean_scores[i]


x_vec = list(task[:-3] for task in tasks  )

rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

lr = 0.01
for task in tasks:
	max_score = -float('Inf')
	file_name = './results/SGD_results_' + task + '_lr_' + str(lr) + '.pkl'
	with open(file_name, 'rb') as f:
		A = pickle.load(f)
		max_score = max(A[1])
		time_totall = sum(A[3])
		hour = int(time_totall/3600.0)
		minute = int( time_totall - 3600 * hour)/60.0
		print('-'*60)
		print('max score for task,'task,' = ',max_score)
		print('time = ', hour, 'hours and ',minute, 'minutes')


lr = 0.00001
for task in tasks:
	max_score = -float('Inf')
	file_name = './results/SGD_results_' + task + '_lr_' + str(lr) + '.pkl'
	with open(file_name, 'rb') as f:
		A = pickle.load(f)
		max_score = max(A[1])
		time_totall = sum(A[3])
		hour = int(time_totall/3600.0)
		minute = int( time_totall - 3600 * hour)/60.0
		print('-'*60)
		print('max score for task,'task,' = ',max_score)
		print('time = ', hour, 'hours and ',minute, 'minutes')

