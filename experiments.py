from play import make_play_env
import numpy as np
import cv2
import os

def run_experiments():
    rams = []
    states = ['start.state', 'mart.state', 'got_parcel.state', 'parcel_delivered.state']
    for state in states:
        env = make_play_env(state)
        env.reset()
        obs, _, terminated, _, info = env.step(np.zeros(9, dtype=np.int8))  # no-op
        print('current_map: ', info['current_map'])
        rams.append(env.get_ram())
        env.close()

    check_start_ram(rams[0])

def check_start_ram(ram):
    startBit = 0xD000
    endBit = 0xDFFF
    startIndex = startBit
    endIndex = endBit
    values = []
    for i in range(startIndex, endIndex):
        values.append(ram[i])

    #check party_count


if __name__ == '__main__':
    run_experiments()