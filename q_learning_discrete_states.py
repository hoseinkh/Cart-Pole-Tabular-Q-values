###############################################################################
# For more info, see https://hoseinkh.github.io/
###############################################################################
# In this file we are going to solve the Cart-Pole problem ...
# ... learning state-action value functions, referred to as Q-values.
#
# This problem has a continuous state space, hence we are going to ...
# ... quanize the state space so that we would deal with discrete states ...
# ... using bins.
# The actions are binary (1: go to right; 0: go to left)
###############################################################################
import gym
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
###############################################################################
def build_state_features(features):
  return tuple(features)
###############################################################################
def to_bin(value, bins):
  index_in_curr_bin = np.digitize(x=[value], bins=bins)[0]
  return index_in_curr_bin
###############################################################################
def calc_index(out_index_list, list_of_bin_sizes):
  out_index = 0
  for i in range(len(list_of_bin_sizes)-1,-1,-1):
    if i == len(list_of_bin_sizes)-1:
      list_of_cumulative_bin_sizes = [1]
    else:
      temp_list = [list_of_cumulative_bin_sizes[0]*list_of_bin_sizes[i+1]]
      temp_list.extend(list_of_cumulative_bin_sizes)
      list_of_cumulative_bin_sizes = temp_list
  # calculating the single index
  for i in range(len(out_index_list)-1,-1,-1):
    out_index += out_index_list[i]*list_of_cumulative_bin_sizes[i]
  return out_index
###############################################################################
class FeatureTransformer:
  def __init__(self):
    # Note: to make this better you could look at how often each bin was
    # actually used while running the script.
    # It's not clear from the high/low values nor sample() what values
    # we really expect to get.
    self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
    self.cart_velocity_bins = np.linspace(-2, 2, 9) # (-inf, inf) (I did not check that these were good values)
    self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
    self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9) # (-inf, inf) (I did not check that these were good values)
  ######################################
  def transform(self, observation):
    # observation: current observed state
    # returns an integer index that we use to find the corresponding Q-value ...
    # ... for this state over all actions. Since we have only two action, ...
    # ... we can find the corresponding Q-value for this state when the action ...
    # ... is 0 (by accessing Model.Q[ind, 0]), or when action is 1 (by ...
    # ... accessing Model.Q[ind, 1]).
    # Hence, this function returns the corresponding index in the Q-table!
    cart_pos, cart_vel, pole_angle, pole_vel = observation
    #
    feature_representaion_of_state = [
      to_bin(cart_pos, self.cart_position_bins),
      to_bin(cart_vel, self.cart_velocity_bins),
      to_bin(pole_angle, self.pole_angle_bins),
      to_bin(pole_vel, self.pole_velocity_bins)
    ]
    list_of_bin_sizes = [len(self.cart_position_bins)+1, len(self.cart_velocity_bins)+1, len(self.pole_angle_bins)+1, len(self.pole_velocity_bins)+1]
    #
    out_index = calc_index(feature_representaion_of_state, list_of_bin_sizes)
    return out_index
###############################################################################
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.feature_transformer = feature_transformer
    #
    num_states = ((len(self.feature_transformer.cart_position_bins) + 1) *
                  (len(self.feature_transformer.cart_velocity_bins) + 1) *
                  (len(self.feature_transformer.pole_angle_bins   ) + 1) *
                  (len(self.feature_transformer.pole_velocity_bins) + 1) )
    # num_states = 10**env.observation_space.shape[0]
    num_actions = env.action_space.n # = 2
    self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
  ######################################
  # s: a state that we want to find its corresponding Q-values.
  # This is pretty much a lookup from the Q-table.
  # This method returns a vector of all the Q-values for all the actions ...
  # ... corresponding to the state s
  def predict(self, s):
    index_of_s_in_Q = self.feature_transformer.transform(s)
    return self.Q[index_of_s_in_Q]
  ######################################
  # s: state
  # a: action
  # G: target return
  def update(self, s, a, G, alpha):
    index_of_state_s = self.feature_transformer.transform(s)
    self.Q[index_of_state_s,a] += alpha*(G - self.Q[index_of_state_s,a])
  ######################################
  # this method selects an action using epsilon-greefy algorithm
  def epsilon_greedy_action_selection(self, s, eps):
    if np.random.random() < eps: # do exploration
      return self.env.action_space.sample()
    else: # do exploitation (be greedy)
      Q_values_for_this_state = self.predict(s)
      return np.argmax(Q_values_for_this_state) # return the action that results in the highest Q-value
###############################################################################
def play_one(model, eps, discount_rate, alpha):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 10000:
    action = model.epsilon_greedy_action_selection(observation, eps)
    prev_observation = observation
    # play a step
    observation, reward, done, info = env.step(action)
    # add reward to the total reward
    totalreward += reward
    #
    if done and iters < 199:
      reward = -300
    #
    # update the model
    G = reward + discount_rate*np.max(model.predict(observation))
    model.update(prev_observation, action, G, alpha)
    #
    iters += 1
  return totalreward
###############################################################################
# we are evaluating the performance of the model at each time t by ...
# ... taking the running average of the adjacent 100 iterations to that time t.
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.xlabel("Iterations")
  plt.ylabel("Average Time")
  # plt.show()
  plt.savefig(curr_path + '/figs/reward_running_avg_CartPole_tabular_state_action_values.png')
  plt.close()
###############################################################################
if __name__ == '__main__':
  ## initialization part
  curr_path = os.path.abspath(os.getcwd())
  env = gym.make('CartPole-v0')
  ft = FeatureTransformer()
  model = Model(env, ft)
  alpha = 0.01
  discount_rate = 0.9
  num_of_episodes = 10000
  ## we can use monitor to record
  if False:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './video/' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
  #
  totalrewards = np.empty(num_of_episodes)
  for n in tqdm(range(num_of_episodes)):
    # we are decaying the value of epsilon
    curr_eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(model, curr_eps, discount_rate, alpha)
    totalrewards[n] = totalreward
    # print progress every 100 steps!
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "curr_eps:", curr_eps)
  #
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())
  #
  plt.plot(totalrewards)
  plt.xlabel("Iterations")
  plt.ylabel("Running Average Time")
  plt.savefig(curr_path + '/figs/reward_avg_CartPole_tabular_state_action_values.png')
  plt.close()
  # plt.show()
  #
  plot_running_avg(totalrewards)
###############################################################################


