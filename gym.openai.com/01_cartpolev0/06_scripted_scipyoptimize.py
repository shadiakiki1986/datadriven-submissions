"""
General
Instead of grid search, use scipy.optimize to search for the optimal parameters

Dev notes
pew new openai_gym
pip install gym pandas (strike: matplotlib)
python 02_scripted.py
"""
#---------------------------

import gym
import pandas as pd
import numpy as np
import time

# Cart Pole
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
env = gym.make('CartPole-v0')

# convenient variable names
action_left = 0
action_right = 1

#-----------------------------------------------------------------

class Controller:

  def __init__(self, obs_limits):
    self.observation_all = self.get_init_observation_all()
    self.obs_limits = obs_limits

  def get_init_observation_all(self):
    observation_all = pd.concat([
        pd.DataFrame(np.zeros(shape=[1000,  7], dtype=float)),
        pd.DataFrame(np.full([1000,  4], np.nan )),
        pd.Series([""]*1000),
      ], axis=1)
    observation_all.columns=[
        "cart_position",
        "cart_velocity",
        "pole_angle", # degrees
        "pole_velocity_at_tip",
        # alert_* is a normalized fraction, 0 being no alert, 1 being "reached the limit before losing", and > 1 meaning that we lost
        "alert_cp",
        "alert_cv",
        "alert_pa",
        # actions for each goal
        "action_pa",
        "action_cp",
        "action_cv",
        # weighted average action
        "action_todo",
        "msg",
      ]
    return observation_all


  def pa_to_action(self, ol_series):
    msg2 = ""

    ## pa_diff = self.observation_all["pole_angle"].tail(n=2)
    #pa_diff = self.observation_all["pole_angle"].loc[[ol_series.name-1,ol_series.name]]
    #pa_diff = pa_diff.diff().values[-1]
    #if (pa_diff * ol_series["pole_angle"]) > 0:
    #    msg2 = " but it's not improving yet"
    #    if ol_series["pole_angle"] < 0:
    #        action_todo = action_left
    #    else:
    #        action_todo = action_right
    #else:
    #    # and it's already improving
    #    action_todo = np.nan

    if ol_series["pole_angle"] < 0:
        action_todo = action_left
    else:
        action_todo = action_right

    return action_todo, msg2


  def cp_to_action(self, ol_series):
    msg2 = ""
    cp_diff = self.observation_all["cart_position"].loc[[ol_series.name-1,ol_series.name]]
    cp_diff = cp_diff.diff().values[-1]
    if (cp_diff * ol_series["cart_position"]) > 0:
        msg2 = " but it's not improving yet"
        if ol_series["cart_position"] < 0:
            action_todo = action_right
        else:
            action_todo = action_left
    else:
      # already improving
      action_todo = np.nan

    return action_todo, msg2


  def cv_to_action(self, ol_series):
    if ol_series["cart_velocity"] < 0:
        action_todo = action_right
    else:
        action_todo = action_left

    return action_todo



  # decide on action from observations
  def ol_to_action(self, ol_series):
    msg = ""
    
    # get actions
    action_pa, msg_pa = self.pa_to_action(ol_series)
    action_cp, msg_cp = self.cp_to_action(ol_series)
    action_cv         = self.cv_to_action(ol_series)

    # get message to display
    if abs(ol_series["alert_pa"]) > 0.5:
        msg1 = "pole angle emergency situation"
        msg = msg1 + " " + msg_pa

    else:
        #print("\tpole angle already under control")
        if abs(ol_series["alert_cp"]) > 0.5:
            msg1 = "cart position emergency situation"
            msg = msg1 + " " + msg_cp

        else:
            if abs(ol_series["alert_cv"]) > 0.5:
                msg = "cart velocity emergency situation"

    # weighted-average of actions
    action_todo = (    0 if pd.isnull(action_pa) else (2*(action_pa - 0.5)) * abs(ol_series['alert_pa'])
                  ) + (0 if pd.isnull(action_cp) else (2*(action_cp - 0.5)) * abs(ol_series['alert_cp'])
                  ) + (0 if pd.isnull(action_cv) else (2*(action_cv - 0.5)) * abs(ol_series['alert_cv'])
                  )
    action_todo = 1 if action_todo >= 0 else 0

    return action_pa, action_cp, action_cv, action_todo, msg

  def observation_last_array_to_dataframe(self, ol_array, current_index):
        # clearer variable names
        cart_position = ol_array[0]
        cart_velocity = ol_array[1]
        pole_angle = ol_array[2] / 3.14 * 180
        pole_velocity_at_tip = ol_array[3]
        msg = ""
        
        # log env
        ol_dict = {
            "cart_position": cart_position,
            "cart_velocity": cart_velocity,
            "pole_angle": pole_angle, # degrees
            "pole_velocity_at_tip": pole_velocity_at_tip,
            "msg": msg,
            "alert_cp": cart_position / self.obs_limits['cart_position'],
            "alert_cv": cart_velocity  / self.obs_limits['cart_velocity'],
            "alert_pa": pole_angle / self.obs_limits['pole_angle'],
            "action_pa": np.nan,
            "action_cp": np.nan,
            "action_cv": np.nan,
            "action_todo": np.nan,
        }
        # observation_all = observation_all.append(ol_dict, ignore_index=True)

        # set values from dict
        for k,v in ol_dict.items():
          self.observation_all.loc[current_index, k] = v

#-----------------------------------------------------------------

# array of thresholds to check iteratively
# Even though the limit for pole angle is 12, need to set the threshold to 6 to maintain control
# Also, cart velocity doesn't have a specific limit, so trial and error choosing 4
# obs_limits = {"cart_position": 2.4, "cart_velocity": 2, "pole_angle": 12, }
# obs_limits = {"cart_position": 2.4, "cart_velocity": 3, "pole_angle": 6, }
obs_limits = pd.DataFrame({
  "cart_position": np.arange(0,5,.1),
  "cart_velocity": np.arange(0,5,.1),
  "pole_angle": np.arange(0,12,.2)[:50],
})

obs_limits = pd.DataFrame(
  np.array(np.meshgrid(
    np.arange(1,5,1),
    np.arange(1,5,1),
    np.arange(1,12,1)[:5],
  )).T.reshape(-1,3),
  columns=["cart_position", "cart_velocity", "pole_angle"]
)
obs_limits['stat'] = np.nan
# obs_limits = obs_limits.head(n=4)

#-----------------------------------------------------------------

def run_game_scoring(settings_ndarray):
  settings_val = {"cart_position": settings_ndarray[0], "cart_velocity": settings_ndarray[1], "pole_angle": settings_ndarray[2]}
  print("run game scoring", settings_val)

  if settings_val["cart_position"] <= 0: return 0
  if settings_val["cart_velocity"] <= 0: return 0
  if settings_val["pole_angle"] <= 0: return 0

  n_games = 1 # 6
  game_score = np.full([n_games], np.nan )
  for game_i in range(game_score.shape[0]):
    print("\tGame", game_i+1, game_score.shape[0])

    env.reset()

    # step in simulation
    ctrl = Controller(settings_val)

    for _ in range(1000):
      #print(_, 1000)
      # env.render()

      # choose action (check docs link above for more details)
      if _ == 0:
          action_todo = env.action_space.sample()
      else:
          #print("\tol_array", ol_array)

          # clearer variable names
          ctrl.observation_last_array_to_dataframe(ol_array, _)

          # get an action to do
          # note that this is a reference, so if I modify any of the keys, the original dataframe will be modified too
          # ol_series = observation_all.iloc[-1]
          ol_series = ctrl.observation_all.iloc[_]

          # get action
          action_pa, action_cp, action_cv, action_todo, msg = ctrl.ol_to_action(ol_series)

          # save
          ctrl.observation_all.loc[_, "action_pa"  ] = action_pa
          ctrl.observation_all.loc[_, "action_cp"  ] = action_cp
          ctrl.observation_all.loc[_, "action_cv"  ] = action_cv
          ctrl.observation_all.loc[_, "msg"        ] = msg

      # take the scripted action
      # ol_array = obervation_last_array (i.e. "last observation in array format")
      ctrl.observation_all.loc[_, "action_todo"] = action_todo
      ol_array, reward, done, info = env.step(action_todo)
      action_last = action_todo

      #import pdb
      #pdb.set_trace()
      if done:
        # print("\t\tgame over", game_i, "/", _, 1000, "/", "You win" if _+1 == 200 else "You lose")
        game_score[game_i] = True if _+1 == 200 else False
        break

    if False:
      print(ctrl.observation_all.head(n=_).tail(n=40))

    if False:
      fn = "04_observation_all.csv"
      ctrl.observation_all.head(n=_+1).to_csv(fn)
      print("saved matrix to ", fn)

  return game_score.sum()

"""
# grid search
for settings_i, settings_val in obs_limits.iterrows():
  print(time.ctime(), "settings", settings_i, obs_limits.shape[0])
  obs_limits.loc[settings_i, "stat"] = run_game_scoring(settings_val["cart_position"], settings_val["cart_velocity"], settings_val["pole_angle"]):

import pdb
pdb.set_trace()

print("best combination is")
# print(obs_limits.loc[obs_limits['stat'].idxmax()]) # doesn't work in multiple max case
print(obs_limits.loc[(obs_limits['stat'] == obs_limits['stat'].max())])

obs_limits.to_csv("05_obs_limits.csv", index=False)
#---------------------------------
#obs_limits_p1 = pd.read_csv("05_obs_limits_20190110T1600_p1.csv")
#obs_limits_both = obs_limits.merge(obs_limits_p1, on=["cart_position", "cart_velocity", "pole_angle"], suffixes=["_p2", "_p1"])
#obs_limits_both["stat"] = obs_limits_both["stat_p1"] + obs_limits_both["stat_p2"]
#print(obs_limits_both.loc[(obs_limits_both['stat'] == obs_limits_both['stat'].max())])
"""


# Test function above
# print(run_game_scoring(np.array([1, 1, 1])))

"""
Turns out this is for floats and not integers

# Scipy minimization docs
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
from scipy.optimize import minimize
result = minimize(
  lambda x: -1*run_game_scoring(x),
  np.array([1, 1, 1]),
  bounds=[(1, 6), (1, 6), (1, 6)],
  #method='SLSQP',
  #options={'disp': True,'ftol':1.0,}
  method='Nelder-Mead',
  options={'xtol': 1e-2, 'disp': True,'ftol':1.0,}

  )
"""

"""
# Try with hyperopt
from hyperopt import hp
space = hp.choice(
    'a',
    [
      np.arange(1,5,1),
      np.arange(1,5,1),
      np.arange(1,12,1)[:5],

       ('case 1', 1 + hp.lognormal('c1', 0, 1)),
       ('case 2', hp.uniform('c2', -10, 10))
    ]
)
"""

"""
# https://nbviewer.jupyter.org/github/cochoa0x1/intro-mathematical-programming/blob/master/01-introduction/Linear%20Programming.ipynb
from pulp import *
prob = LpProblem("Hello-Mathematical-Programming-World!",LpMinimize)
x = LpVariable('x',lowBound=0, upBound=10, cat='Continuous')
y = LpVariable('y',lowBound=0, upBound=10, cat='Continuous')
z = LpVariable('z',lowBound=0, upBound=10, cat='Continuous')
objective = -1*run_game_scoring(np.array([x,y,z]))
prob += objective
#prob.solve()
#print(LpStatus[prob.status])
"""

# Scipy minimization docs
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.basinhopping.html
from scipy.optimize import basinhopping

class MyBounds(object):
    def __init__(self, xmax=[1.1,1.1,1.1], xmin=[-1.1,-1.1,-1.1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

mybounds = MyBounds([6,6,6], [1,1,1])
minimizer_kwargs = {"method": "BFGS"}

result = basinhopping(
  lambda x: -1*run_game_scoring(x),
  x0 = np.array([1, 1, 1]),
  minimizer_kwargs=minimizer_kwargs,
  stepsize = 1,
  accept_test=mybounds
  )

print("global minimum: x = [%.4f, %.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
                                                                ret.x[1],
                                                                ret.x[2],
                                                                ret.fun))

import pdb
pdb.set_trace()
