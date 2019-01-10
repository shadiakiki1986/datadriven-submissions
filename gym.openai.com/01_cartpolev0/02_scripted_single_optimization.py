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
    "th_i": [],
})

for _ in range(1000):
    print(_, 1000)
    # env.render()

    # choose action (check docs link above for more details)
    if _ == 0:
        action_todo = env.action_space.sample()
    else:
        #print("\tobservation_last", observation_last)

        cart_position = observation_last[0]
        cart_velocity = observation_last[1]
        pole_angle = observation_last[2] / 3.14 * 180
        pole_velocity_at_tip = observation_last[3]
        action_left = 0
        action_right = 1
        action_todo = np.nan
        msg = ""
        
        thresholds_all = [
            {"pole_angle": 8, "cart_position": 4, "cart_velocity": 2},
            {"pole_angle": 5, "cart_position": 2, "cart_velocity": 1},
        ]

        for th_i, th_val in enumerate(thresholds_all):
            if pd.notnull(action_todo): break

            if abs(pole_angle) > th_val["pole_angle"]:
                msg = "pole angle emergency situation"
                pa_diff = observation_all["pole_angle"].tail(n=2).diff().values[-1]
                if (pa_diff * pole_angle) > 0:
                    msg += " but it's not improving yet"
                    if pole_angle < 0:
                        action_todo = action_left
                    else:
                        action_todo = action_right
                else:
                    # and it's improving
                    if abs(cart_velocity) > 1:
                        msg += " and cart is too fast"
                        if cart_velocity > 0:
                            action_todo = action_left
                        else:
                            action_todo = action_right
                    else:
                        if pole_angle < 0:
                            action_todo = action_left
                        else:
                            action_todo = action_right
            else:
                #print("\tpole angle already under control")
                if abs(cart_position) > th_val["cart_position"]:
                    msg = "cart position emergency situation"
                    if cart_position < 0:
                        action_todo = action_right
                    else:
                        action_todo = action_left
                else:
                    if abs(cart_velocity) > th_val["cart_velocity"]:
                        msg = "cart velocity emergency situation"
                        if cart_velocity < 0:
                            action_todo = action_right
                        else:
                            action_todo = action_left
                    else:
                        #print("\tcart velocity under control")
                        if abs(cart_position) > 0.5:
                            msg = "cart position not in center"
                            if abs(cart_velocity) > 0.3:
                                msg += " but cart velocity not slow"
                                if cart_velocity < 0:
                                    action_todo = action_right
                                else:
                                    action_todo = action_left
                            else:
                                # cart position is close to center
                                if cart_position * cart_velocity > 0:
                                    msg += "but we're not in the correct direction"
                                    if cart_velocity > 0:
                                        action_todo = action_left
                                    else:
                                        # and we're already in the right direction
                                        if action_last == action_left:
                                            action_todo = action_right
                                        else:
                                            action_todo = action_left
                                else:
                                    # and we're already in the correct direction
                                    if action_last == action_left:
                                        action_todo = action_right
                                    else:
                                        action_todo = action_left

                        else:
                            if th_i == (len(thresholds_all)-1):
                                # msg = "just sway with me"
                                if action_last == action_left:
                                    action_todo = action_right
                                else:
                                    action_todo = action_left
                            else:
                                # change thresholds and check again
                                action_todo = np.nan


        # log env
        observation_all = observation_all.append({
            "cart_position": cart_position,
            "cart_velocity": cart_velocity,
            "pole_angle": pole_angle, # degrees
            "pole_velocity_at_tip": pole_velocity_at_tip,
            "action_todo": action_todo,
            "msg": msg,
            "th_i": th_i,
        }, ignore_index=True)

    # take the scripted action
    observation_last, reward, done, info = env.step(action_todo)
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
