# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A A2C agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import environment

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NO_OP = actions.FUNCTIONS.no_op.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class A2CAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def __init__(self):
    super(A2CAgent, self).__init__()
    self._t_max = 40
    self._lr = 5e-4

  def setup(self, obs_spec, action_spec):
    super(A2CAgent, self).setup(obs_spec, action_spec)
    self._model = FullyConvNet(self.obs_spec["screen"][1])
    self._optimizer = optim.RMSprop(self._model.parameters(), lr=self._lr)

  def step(self, obs):
    super(A2CAgent, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      obs_feat = obs.observation["screen"][_PLAYER_RELATIVE].astype(np.float32)
      obs_feat = torch.from_numpy(obs_feat).unsqueeze(0).unsqueeze(0)
      if self._last_policy is not None:
        self._learn(last_policy=self._last_policy,
                    last_value=self._last_value,
                    last_action=self._last_action,
                    reward=obs.reward,
                    discount=obs.discount,
                    obs_feat=obs_feat,
                    done=obs.step_type == environment.StepType.LAST)

      policy, value = self._model(Variable(obs_feat))
      action = policy.multinomial(1)

      self._last_policy = copy(policy)
      self._last_value = copy(value)
      self._last_action = Variable(action.data)

      action_xy = np.unravel_index(action.data.numpy(),
                                   self.obs_spec["screen"][1:3])
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, action_xy])
    else:
      print("select")
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

  def reset(self):
    super(A2CAgent, self).reset()
    self._rewards = []
    self._values = []
    self._log_actions = []
    self._entropies = []
    self._last_policy, self._last_value, self._last_action = None, None, None

  def _learn(self, last_policy, last_value, last_action,
             reward, discount, obs_feat, done):
    log_policy = torch.log(last_policy)
    log_action = log_policy.gather(1, last_action)
    entropy = -(log_policy * last_policy).sum(1)

    self._rewards.append(reward)
    self._values.append(last_value)
    self._log_actions.append(log_action)
    self._entropies.append(entropy)

    if done or len(self._rewards) >= self._t_max:
      r = torch.zeros(1, 1)
      if not done:
        _, value = self._model(Variable(obs_feat))
        r = value.data
      self._values.append(Variable(r))
      r = Variable(r)
      gae = torch.zeros(1, 1)

      # compute n-step loss
      value_loss, policy_loss = 0, 0
      for i in reversed(range(len(self._rewards))):
        r = discount * r + self._rewards[i]
        advantage = r - self._values[i]
        value_loss += 0.5 * advantage.pow(2)
        # Generalized Advantage Estimataion
        delta_t = self._rewards[i] + discount * \
          self._values[i + 1].data - self._values[i].data
        gae = gae * discount + delta_t
        policy_loss -= self._log_actions[i] * Variable(gae) \
          + 0.01 * self._entropies[i]

      # compute grad and optimize
      self._optimizer.zero_grad()
      loss = policy_loss + 0.5 * value_loss
      loss.backward()
      torch.nn.utils.clip_grad_norm(self._model.parameters(), 40)
      self._optimizer.step()

      self._rewards = []
      self._values = []
      self._log_actions = []
      self._entropies = []


class FullyConvNet(nn.Module):
    def __init__(self, screen_size):
        super(FullyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,
                               stride=1, padding=0)
        self.fc = nn.Linear(32 * screen_size * screen_size, 256)
        self.value_fc = nn.Linear(256, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        v = F.relu(self.fc(x.view(x.size(0), -1)))
        value = self.value_fc(v)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        policy = self.softmax(x)
        return policy, value
