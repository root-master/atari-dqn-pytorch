# create the environment
# task = 'Breakout-v0'
# task = 'BeamRider-v0'
# task = 'Enduro-v0'
# task = 'Pong-v0'
# task = 'Qbert-v0'
# task = 'Seaquest-v0'
# task = 'SpaceInvaders-v0'

parser = argparse.ArgumentParser(description='Quasi-Newton DQN feat. PyTorch')
parser.add_argument('--batch','-batch', type=int, default=32, metavar='b',
                    help='input batch size for training')
parser.add_argument('--task','-task', type=str, default='Breakout-v0', metavar='T',
                    help='choose an ATARI task to play')
parser.add_argument('--optim','-optim', type=str, default='RMSProp', metavar='T',
                    help='choose an ATARI task to play')

parser.add_argument('--lr','-lr', type=float, default=0.00025, metavar='T',
                    help='choose an ATARI task to play')

parser.add_argument('--max-iter','-maxiter', type=int, default=2000*1024, metavar='max-iter',
                    help='max steps for Deep RL algorithm')

args = parser.parse_args()
task = args.task
batch_size = int(args.batch)
lr = float(args.lr)
optim = args.optim
max_iter = int(args.max_iter)


from Environment import Environment
env = Environment(task=task) 
num_actions = env.env.action_space.n
print('number of actions: ', num_actions)

# create expereince memory
from memory import ExperienceMemory 
experience_memory = ExperienceMemory() 

# create agent 
from agent import Controller
controller = Controller(experience_memory=experience_memory,
						num_actions=num_actions,
						lr=lr,
						batch_size=batch_size,
						optim_method=optim) 

# create the trainer
from trainer import Trainer
atari_trainer = Trainer(env=env,
				 		controller=controller,
				 		experience_memory=experience_memory,
				 		max_iter=max_iter)

# run the training loop
atari_trainer.train()