# pew new openai_gym
# pip install gym pandas (strike: matplotlib)
# python 02_scripted.py
#---------------------------

import gym
import pandas as pd
import numpy as np
import time

# Cart Pole
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
env = gym.make('CartPole-v0')
env.reset()

# get observation
# (Pdb) print(env.observation_space.high)
# [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
# (Pdb) print(env.observation_space.low)
# [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

#observation_all = pd.DataFrame({
#    "cart_position": [],
#    "cart_velocity": [],
#    "pole_angle": [], # degrees
#    "pole_velocity_at_tip": [],
#    # alert_* is a normalized fraction, 0 being no alert, 1 being "reached the limit before losing", and > 1 meaning that we lost
#    "alert_cp": [],
#    "alert_cv": [],
#    "alert_pa": [],
#    # actions for each goal
#    "action_pa": [],
#    "action_cp": [],
#    "action_cv": [],
#    # weighted average action
#    "action_todo": [],
#    # english summary of row
#    "msg": [],
#})
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

# array of thresholds to check iteratively
# Even though the limit for pole angle is 12, need to set the threshold to 6 to maintain control
# Also, cart velocity doesn't have a specific limit, so trial and error choosing 4
# obs_limits = {"cart_position": 2.4, "cart_velocity": 2, "pole_angle": 12, }
obs_limits = {"cart_position": 2.4, "cart_velocity": 3, "pole_angle": 6, }

# convenient variable names
action_left = 0
action_right = 1

def pa_to_action(observation_all, ol_series):
  msg2 = ""

  ## pa_diff = observation_all["pole_angle"].tail(n=2)
  #pa_diff = observation_all["pole_angle"].loc[[ol_series.name-1,ol_series.name]]
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


def cp_to_action(observation_all, ol_series):
  msg2 = ""
  cp_diff = observation_all["cart_position"].loc[[ol_series.name-1,ol_series.name]]
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


def cv_to_action(observation_all, ol_series):
  if ol_series["cart_velocity"] < 0:
      action_todo = action_right
  else:
      action_todo = action_left

  return action_todo



# decide on action from observations
def ol_to_action(observation_all, ol_series):
  msg = ""
  
  # get actions
  action_pa, msg_pa = pa_to_action(observation_all, ol_series)
  action_cp, msg_cp = cp_to_action(observation_all, ol_series)
  action_cv         = cv_to_action(observation_all, ol_series)

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


# step in simulation
for _ in range(1000):
    #print(_, 1000)
    env.render()

    # choose action (check docs link above for more details)
    if _ == 0:
        action_todo = env.action_space.sample()
    else:
        #print("\tol_array", ol_array)

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
            "alert_cp": cart_position / obs_limits['cart_position'],
            "alert_cv": cart_velocity  / obs_limits['cart_velocity'],
            "alert_pa": pole_angle / obs_limits['pole_angle'],
            "action_pa": np.nan,
            "action_cp": np.nan,
            "action_cv": np.nan,
            "action_todo": np.nan,
        }
        # observation_all = observation_all.append(ol_dict, ignore_index=True)

        # set values from dict
        for k,v in ol_dict.items():
          observation_all.loc[_, k] = v

        # get an action to do
        # note that this is a reference, so if I modify any of the keys, the original dataframe will be modified too
        # ol_series = observation_all.iloc[-1]
        ol_series = observation_all.iloc[_]

        # get action
        action_pa, action_cp, action_cv, action_todo, msg = ol_to_action(observation_all, ol_series)

        # save
        observation_all.loc[_, "action_pa"  ] = action_pa
        observation_all.loc[_, "action_cp"  ] = action_cp
        observation_all.loc[_, "action_cv"  ] = action_cv
        observation_all.loc[_, "msg"        ] = msg

    # take the scripted action
    # ol_array = obervation_last_array (i.e. "last observation in array format")
    observation_all.loc[_, "action_todo"] = action_todo
    ol_array, reward, done, info = env.step(action_todo)
    action_last = action_todo

    #import pdb
    #pdb.set_trace()
    if done:
        print("game over", _, 1000)
        print("You win" if _+1 == 200 else "You lose")
        break

if False:
  print(observation_all.head(n=_).tail(n=40))

if False:
  fn = "03_observation_all.csv"
  observation_all.head(n=_+1).to_csv(fn)
  print("saved matrix to ", fn)


#from matplotlib import pyplot as plt
#observation_all.plot()
#plt.show()
