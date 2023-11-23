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
    def init(self, env):
        self.env = env

    def encode(self, state):
        return state

    @property
    def size(self):
        return self.env.observation_space.shape[0]


class RBFFeatureEncoder:
    def init(self, env, n_components=100, gamma=0.1):
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




def update_transition(self, s, action, s_prime, reward, done):
    s_feats = self.feature_encoder.encode(s)
    s_prime_feats = self.feature_encoder.encode(s_prime)
    delta = reward + self.gamma * np.max(self.Q(s_prime_feats)) - self.Q(s_feats)[action]
    
    
    self.traces *= self.gamma * self.lambda_
    self.traces[action] += s_feats

    self.weights[action] += self.alpha * delta * self.traces[action]  
    if done: self.traces = np.zeros(self.traces.shape)

