import gym
import numpy as np
import matplotlib.pyplot as plt


# Hyperparameters
ALPHA = 0.25
GAMMA = 0.9
NUM_EPISODES = 10

# Display visualisation and extra information
SLOW = False

# values ranging from -1.2 to 0.6, altered by 0.1
Q_VALUE_POSITIONS = 19

# values ranging from -0.07 to 0.07, altered by 0.01
Q_VALUE_SPEEDS = 15

# 3 actions (0, 1 nad 2) available
Q_VALUE_ACTIONS = 3

# Rounding to decimal places
POSITION_ROUNDING_PLACES = 1
SPEED_ROUNDING_PLACES = 2

# Index shifting
POSITION_SHIFT_TO_0 = 12
SPEED_SHIFT_TO_0 = 7

# create and initialize Q-values tensor
q_values = np.zeros((Q_VALUE_POSITIONS, Q_VALUE_SPEEDS, Q_VALUE_ACTIONS))


def discretize_observation(raw_position, raw_speed):
	# round values to one/two digits after comma
    rounded_position = round(raw_position, POSITION_ROUNDING_PLACES)
    rounded_speed = round(raw_speed, SPEED_ROUNDING_PLACES)

	# tuen the discretized values into index numbers (which start from 0)
    index_position = (rounded_position * np.power(10, POSITION_ROUNDING_PLACES) + POSITION_SHIFT_TO_0)
    index_speed = (rounded_speed * np.power(10, SPEED_ROUNDING_PLACES) + SPEED_SHIFT_TO_0)

    return int(index_position), int(index_speed)
	
	
def determine_action(state):
	position, speed = state
	
	#discretize the state
	position_index, speed_index = discretize_observation(position, speed)
	
	# the actions for this problem are 0, 1 or 2
	best_action = np.argmax(q_values[position_index, speed_index, :])
	
	return best_action
	
	
def update_q_values(state, action, next_state, reward):
	# current state
	position, speed = state
	
	#discretize current state
	position_index, speed_index = discretize_observation(position, speed)
	
	# next state
	next_position, next_speed = next_state
	
	#discretize next state
	next_position_index, next_speed_index = discretize_observation(next_position, next_speed)
	
	max_q_for_next_state = np.max(q_values[next_position_index, next_speed_index, :])
	
	q_values[position_index, speed_index, action] = (1 - ALPHA) * q_values[position_index, speed_index, action] + ALPHA * (reward + GAMMA * max_q_for_next_state)
	
	
def main():	
	# Increase the max number of episodes
	gym.envs.register(
		id='MountainCarMyEasyVersion-v0',
		entry_point='gym.envs.classic_control:MountainCarEnv',
		max_episode_steps=100000,      # MountainCar-v0 uses 200
	)
	env = gym.make('MountainCarMyEasyVersion-v0')

	for _ in range(NUM_EPISODES):
		observation = env.reset()
		done = False
		timesteps = 0
		while not done:
			if SLOW:
				env.render()

			action = determine_action(observation)
			next_observation, reward, done, info = env.step(action)
			update_q_values(observation, action, next_observation, reward)
			observation = next_observation
			
			timesteps += 1
			if SLOW: 
				print (observation)
			if SLOW: 
				print (reward)
			if SLOW: 
				print (done)

		print ("Episode finished after ", timesteps, "timesteps.")

	# Extract the State Value Function as highest Q-value per given postion-speed couple observation
	state_value_function = np.min(q_values, axis=2)
		
	# Plot the State Value Function
	plt.imshow(state_value_function.T)
	plt.colorbar()
	plt.xlabel("Discretized positions")
	plt.ylabel("Discretized speed")
	plt.title("State Value Function")
	plt.show() 
	
	
if __name__ == '__main__':
	main()
	
