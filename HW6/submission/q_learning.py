from environment import MountainCar
import numpy as np 
import sys


def sparse_dot(W,x):
    product = 0.0
    for key in x.keys():
        product += x[key]*W[key]
    return product
        

def policy(W,b,s,epsilon):
    
    Q = []
    for action in [0,1,2]:
        Q.append( sparse_dot(W[:,action], s) + b)
    
    if np.random.rand(0,1) < epsilon:
        action = np.random.randint(0,3)
    else:
        action = Q.index( max(Q) )
    return action
        
def train(W,b,action,reward,s1,s2,gamma,lr):
    Q1 = sparse_dot(W[:,action], s1) + b  # value function at (s,a;W)
    
    Q2_action = []
    for action_ in [0,1,2]:
        Q2_action.append(  gamma*( sparse_dot( W[:,action_], s2) + b ) )
    Q2 = reward + max(Q2_action)    
    
    for key in s1.keys():
        W[key][action] -= lr*(Q1-Q2)*s1[key]
    
    b -= lr*(Q1-Q2)
    return W, b

if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int( sys.argv[4] )
    max_iterations = int( sys.argv[5] )
    epsilon  = float( sys.argv[6] )
    gamma = float(sys.argv[7])
    lr = float(sys.argv[8])


    Car = MountainCar(mode)
    Car.state_space
    W = np.zeros( (Car.state_space, 3))

    b = 0
    done = False 
    
    reward_return = []

    for i in range(episodes):
    	total_reward = 0.0
    	s1 = Car.reset()
    	for j in range(max_iterations):
    		action = policy(W,b,s1,epsilon)
    		s2, reward, done = Car.step(action)
    		total_reward += reward
    		W,b = train(W,b,action,reward,s1,s2,gamma,lr)
    		if done:
    			break
    		s1 = s2 
    	reward_return.append( total_reward )


    with open(returns_out, 'w') as f:
    	for val_reward in reward_return:
    		f.write('%.1f \n'%( val_reward) )


    # with open(weight_out,'w') as f:
    # 	f.write( str(b) + '\n')
    # 	for i in range(Car.state_space):
    # 		for j in range(3):
    # 			f.write( str(W[i][j]) + '\n')