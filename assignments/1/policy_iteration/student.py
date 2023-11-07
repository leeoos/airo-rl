import numpy as np

# The end-state is always at [env_size-1,env_size-1].
def reward_function(s, env_size):
    return 1 if np.all(s == np.array([env_size-1, env_size-1])) else 0

# do not modify this function
def reward_probabilities(env_size):
    rewards = np.zeros((env_size*env_size))
    i = 0
    for r in range(env_size):
        for c in range(env_size):
            state = np.array([r,c], dtype=np.uint8)
            rewards[i] = reward_function(state, env_size)
            i+=1

    return rewards

# Check feasibility of the new state.
# If it is a possible state return s_prime, otherwise return s
def check_feasibility(s_prime, s, env_size, obstacles):
  if np.any(s_prime < 0) or np.any(s_prime >= env_size) or obstacles[s_prime[0], s_prime[1]]:
      return s
  return s_prime

def transition_probabilities(env, s, a, env_size, directions, obstacles):
    prob_next_state = np.zeros((env_size, env_size))
    
    # For each possible next state s_prime, added probability of ending up 
    # in that state in the prob_next_state matrix.
    for i in [0, 1, -1]:
        s_prime = s + directions[(a+i)%4]
        s_prime = check_feasibility(s_prime, s, env_size, obstacles)
        prob_next_state[s_prime[0], s_prime[1]] += 1/3 


    return prob_next_state