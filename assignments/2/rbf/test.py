import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env

    def encode(self, state):
        return state

    @property
    def size(self):
        return self.env.observation_space.shape[0]


class RBFFeatureEncoder:
    def __init__(self, env, n_components=100, gamma=0.1):
        self.env = env
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.rbf_encoder = RBFSampler(gamma=gamma, n_components=n_components)
        # Scale features to [0, 1]
        self.scaler = sklearn.preprocessing.StandardScaler()
        # Fit the scaler with an initial state to set its parameters
        #initial_state = np.zeros(self.env.observation_space.shape[0])
        self.scaler.fit(observation_examples)
        self.rbf_encoder.fit(self.scaler.transform(observation_examples))


    def encode(self, state):
        scaled_state = self.scaler.transform([state])
        #self.rbf_encoder.fit(scaled_state)
        return self.rbf_encoder.transform(scaled_state).flatten()

    @property
    def size(self):
        return self.rbf_encoder.n_components

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1,
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9):
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_

    def Q(self, feats):
        feats = feats.reshape(-1, 1)
        return self.weights @ feats

    def update_transition(self, s, action, s_prime, reward, done):
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)

        delta = reward + self.gamma * np.max(self.Q(s_prime_feats)[:]) - self.Q(s_feats)[action]
  
        I_st = 1 if np.array_equal(s, s_prime) else 0


        #if np.array_equal(s, s_prime):
        #   print("lool")
        #q_gradient = s_prime_feats.reshape(-1, 1)
        #print("traces before update:", self.traces.shape)
        self.traces = self.gamma * self.lambda_ * self.traces + I_st
         # Compute the gradient of Q with respect to the weights
        
        #self.traces *= self.gamma * self.lambda_
        #self.traces[action] += s_feats
        #print("traces after update:", self.traces.shape)
        #print("self.weights[action] afrbf_encoderter update:", self.weights[action].shape)
        #print("delta after update:", delta.shape)
        #print("self.traces after update:", self.traces.shape)
        self.weights += self.alpha * delta[0] * self.traces

        #print("Shapes after update:") self.gamma * self.lambda_ * self.traces + I_st
        #print("self.weights.shape:", self.weights.shape)
        #print("self.traces.shape:", self.traces.shape)
        if done:
            self.traces = np.zeros(self.shape)