from model import DQN
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.transforms as T

class Controller():
	def __init__(self,
				 experience_memory=None,
				 num_actions=4,
				 lr = 0.00025,
				 alpha=0.95,
				 eps=0.01,
				 batch_size=32,
				 gamma=0.99,
				 load_pretrained=False,
				 saved_model_path='./models/a.model',
				 optim_method='RMSprop'):
		
		self.experience_memory = experience_memory # expereince replay memory
		self.lr = lr # learning rate
		self.alpha = alpha # optimizer parameter
		self.eps = 0.01 # optimizer parameter
		self.gamma = 0.99
		self.num_actions = num_actions	
		# BUILD MODEL 
		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")

		dfloat_cpu = torch.FloatTensor
		dfloat_gpu = torch.cuda.FloatTensor

		dlong_cpu = torch.LongTensor
		dlong_gpu = torch.cuda.LongTensor

		duint_cpu = torch.ByteTensor
		dunit_gpu = torch.cuda.ByteTensor 
		
		dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
		duinttype = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

		self.dtype = dtype
		self.dlongtype = dlongtype
		self.duinttype = duinttype

		Q = DQN(in_channels=4, num_actions=num_actions).type(dtype)		
		if load_pretrained:			
			Q.load_state_dict(torch.load(saved_model_path))
		Q_t = DQN(in_channels=4, num_actions=num_actions).type(dtype)
		Q_t.load_state_dict(Q.state_dict())
		Q_t.eval()
		for param in Q_t.parameters():
			param.requires_grad = False

		Q = Q.to(self.device)
		Q_t = Q_t.to(self.device)

		if torch.cuda.device_count() > 1:
			Q = nn.DataParallel(Q).to(self.device)
			Q_t = nn.DataParallel(Q_t).to(self.device)

		self.batch_size = batch_size
		self.Q = Q
		self.Q_t = Q_t
		# optimizer
		if optim_method == 'SGD':
			optimizer = torch.optim.SGD(Q.parameters(), lr=self.lr)
		elif optim_method == 'RMSprop':
			optimizer = optim.RMSprop(Q.parameters(),lr=lr, alpha=alpha, eps=eps)
		else:
			optimizer = optim.RMSprop(Q.parameters(),lr=lr, alpha=alpha, eps=eps)
		self.optimizer = optimizer
		print('init: Controller --> OK')

	def get_best_action(self,s):
		q_np = self.compute_Q(s)
		return q_np.argmax()

	def compute_Q(self,s):
		x = s.reshape((1,4,84,84))
		q = self.Q.forward(torch.Tensor(x).type(self.dtype)/255.0)
		q_np = q.cpu().detach().numpy()
		return q_np

	def update_w(self):
		states, actions, rewards, state_primes, dones = \
			self.experience_memory.sample(batch_size=self.batch_size)
		x = torch.Tensor(states).type(self.dtype)	
		xp = torch.Tensor(state_primes).type(self.dtype)
		actions = torch.Tensor(actions).type(self.dlongtype)
		rewards = torch.Tensor(rewards).type(self.dtype)
		dones = torch.Tensor(dones).type(self.dtype)
		# sending data to gpu
		if torch.cuda.is_available():
			with torch.cuda.device(0):
				x = x.to(self.device)
				xp = xp.to(self.device)
				actions = actions.to(self.device)
				rewards = rewards.to(self.device)
				dones = dones.to(self.device)
		# forward path
		q = self.Q.forward(x/255.0)
		q = q.gather(1, actions.unsqueeze(1))
		q = q.squeeze()
		
		q_p1 = self.Q.forward(xp/255.0)
		_, a_prime = q_p1.max(1)

		q_t_p1 = self.Q_t.forward(xp)
		q_t_p1 = q_t_p1.gather(1, a_prime.unsqueeze(1))
		q_t_p1 = q_t_p1.squeeze()

		target = rewards + self.gamma * (1 - dones) * q_t_p1
		error = target - q
		clipped_error = -1.0 * error.clamp(-1, 1)
		
		self.optimizer.zero_grad()
		q.backward(clipped_error.data)
		
		# We can use Huber loss for smoothness
		# loss = F.smooth_l1_loss(q, target)
		# loss.backward()
		
		for param in self.Q.parameters():
			param.grad.data.clamp_(-1, 1)
		
		# update weights
		self.optimizer.step()

	def update_target_params(self):
		self.Q_t.load_state_dict(self.Q.state_dict())

	def save_model(self, model_save_path):
		torch.save(self.Q.state_dict(), model_save_path)
