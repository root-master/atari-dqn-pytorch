# create the environment
# task = 'Breakout-v0'
# task = 'BeamRider-v0'
# task = 'Enduro-v0'
# task = 'Pong-v0'
# task = 'Qbert-v0'
# task = 'Seaquest-v0'
task = 'SpaceInvaders-v0'
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
						num_actions=num_actions) 

# create the trainer
from trainer import Trainer
atari_trainer = Trainer(env=env,
				 		controller=controller,
				 		experience_memory=experience_memory)

# run the training loop
atari_trainer.train()