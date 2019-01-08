# pew new openai_gym
# pip install gym pandas (strike: matplotlib)
# python 02_scripted.py
#---------------------------

import gym
import pandas as pd
import numpy as np

# Cart Pole
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
env = gym.make('CartPole-v0')
env.reset()

# get observation
# (Pdb) print(env.observation_space.high)
# [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
# (Pdb) print(env.observation_space.low)
# [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

observation_all = pd.DataFrame({
    "cart_position": [],
    "cart_velocity": [],
    "pole_angle": [], # degrees
    "pole_velocity_at_tip": [],
    "msg": [],
    # alert_* is a normalized fraction, 0 being no alert, 1 being "reached the limit before losing", and > 1 meaning that we lost
    "alert_cp": [],
    "alert_cv": [],
    "alert_pa": [],
    # actions for each goal
    "action_pa": [],
    "action_cp": [],
    "action_cv": [],
    # weighted average action
    "action_todo": [],
})

# array of thresholds to check iteratively
obs_limits = {"cart_position": 2.4, "cart_velocity": 2, "pole_angle": 12, }

# convenient variable names
action_left = 0
action_right = 1

def pa_to_action(observation_all):
  ol_series = observation_all.iloc[-1] # note that this is a reference, so if I modify any of the keys, the original dataframe will be modified too

  msg2 = ""
  pa_diff = observation_all["pole_angle"].tail(n=2).diff().values[-1]
  if (pa_diff * ol_series["pole_angle"]) > 0:
      msg2 = " but it's not improving yet"
      if ol_series["pole_angle"] < 0:
          action_todo = action_left
      else:
          action_todo = action_right
  else:
      # and it's already improving
      action_todo = np.nan

  return action_todo, msg2


def cp_to_action(observation_all):
  ol_series = observation_all.iloc[-1] # note that this is a reference, so if I modify any of the keys, the original dataframe will be modified too

  msg2 = ""
  cp_diff = observation_all["cart_position"].tail(n=2).diff().values[-1]
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


def cv_to_action(observation_all):
  ol_series = observation_all.iloc[-1] # note that this is a reference, so if I modify any of the keys, the original dataframe will be modified too

  if ol_series["cart_velocity"] < 0:
      action_todo = action_right
  else:
      action_todo = action_left

  return action_todo



# decide on action from observations
def ol_to_action(observation_all):
  ol_series = observation_all.iloc[-1] # note that this is a reference, so if I modify any of the keys, the original dataframe will be modified too
  msg = ""
  
  # get actions
  action_pa, msg_pa = pa_to_action(observation_all)
  action_cp, msg_cp = cp_to_action(observation_all)
  action_cv         = cv_to_action(observation_all)

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
        observation_all = observation_all.append(ol_dict, ignore_index=True)

        # get an action to do
        action_pa, action_cp, action_cv, action_todo, msg = ol_to_action(observation_all)
        observation_all["action_pa"].iloc[-1] = action_pa
        observation_all["action_cp"].iloc[-1] = action_cp
        observation_all["action_cv"].iloc[-1] = action_cv
        observation_all["action_todo"].iloc[-1] = action_todo
        observation_all["msg"].iloc[-1] = msg

    # take the scripted action
    # ol_array = obervation_last_array (i.e. "last observation in array format")
    ol_array, reward, done, info = env.step(action_todo)
    action_last = action_todo

    #import pdb
    #pdb.set_trace()
    if done:
        print("game over", _, 1000)
        break

print(observation_all.tail(n=40))
#from matplotlib import pyplot as plt
#observation_all.plot()
#plt.show()
