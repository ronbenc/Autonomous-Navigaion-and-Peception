from msilib.schema import Class
from tkinter.messagebox import NO
from turtle import st
import numpy as np

class BMDPscenaraio:
    def __init__(self, belief_mean, belief_cov, F, process_cov, beacons, actions, d, rmin, lambda_reg = 0):
        self.belief_mean = belief_mean
        self.belief_cov = belief_cov
        self.F = F
        self.process_cov = process_cov
        self.beacons = beacons # 2XN
        self.actions = actions
        self.d = d
        self.rmin=rmin
        self.lambda_reg = lambda_reg
        self.last_obs_cov = None

    def _generate_observation_from_beacons(self, state):
        distances_from_beacons = np.linalg.norm(self.beacons-state, axis=1)
        min_dist = np.min(distances_from_beacons)
        if min_dist <= self.d:
            obs_cov = 0.01*max(min_dist, self.rmin)*np.eye(2)
            self.last_obs_cov = obs_cov
            obs = state + np.random.multivariate_normal(np.zeros_like(state), obs_cov)
            return obs
        else:
            return None
    

    def _propogate_belief(self, action):
        self.belief_mean = (self.F @ self.belief_mean) + action
        self.belief_cov = (self.F @ self.belief_cov @ self.F.T) + self.process_cov

 
    def _update_belief(self, obs):
        assert self.last_obs_cov is not None
        k = self.belief_cov @ np.linalg.inv(self.belief_cov + self.last_obs_cov)
        self.belief_mean = self.belief_mean + (k @ (obs-self.belief_mean))
        self.belief_cov = (np.eye(2) - k) @ self.belief_cov

    def sample_motion_model(self, action, state):
        noise = np.random.multivariate_normal(np.zeros_like(state), self.process_cov)
        return state + (self.F @ action) + noise

    def transit_belief_MDP(self, action):
        # propogate belief using action and transition model (KF)
        self._propogate_belief(action)

        # sample a state from propogated belief, create observation and if not null update belief
        sampled_state = np.random.multivariate_normal(self.belief_mean, self.belief_cov)
        obs = self._generate_observation_from_beacons(sampled_state)
        if obs is not None:
            self._update_belief(obs)

        return obs

    def _cost(self, goal_state):
        cost = np.linalg.norm(self.belief_mean - goal_state) + self.lambda_reg*np.linalg.det(self.belief_cov)
        return cost


# class BMDP_planner:
#     def __init__(self, goal_state, number_of_samples) -> None:
#         self.goal_state = goal_state
#         self.number_of_samples = number_of_samples

    
    def sparse_sampling(self, goal_state, number_of_samples, horizon):
        if horizon == 0:
            return None, self._cost(goal_state)
        best_action, best_value = None, np.inf
        curr_state = self.belief_mean, self.belief_cov
        cost = self._cost(goal_state)  # does not depend on action in our case
        for action in self.actions:
            # compute average value over n samples for current action
            avg_future_val = 0
            for i in range(number_of_samples):
                self.transit_belief_MDP(action)
                selected_action, value = self.sparse_sampling(goal_state, number_of_samples, horizon-1)
                avg_future_val += value
                self.belief_mean, self.belief_cov = curr_state
            
            #update action and value if they are better
            future_val_for_action = avg_future_val/number_of_samples
            if future_val_for_action < best_value:
                best_action, best_value = action, future_val_for_action

        return best_action, best_value + cost


