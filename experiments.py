from envs import make_play_env
import numpy as np

workRamStartHex = 0xD000
ramStartHex = 0x8000
    
def run_experiments():
    states = ['start.state', 'mart.state', 'got_parcel.state', 'parcel_delivered.state']
    rams = []
    for state in states:
        env = make_play_env(state)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(9, dtype=np.int8))  # no-op
        currentRam = env.get_ram()
        check_start_ram(info, currentRam)
        rams.append(currentRam)
        env.close()
        
    print_diffs(rams)
        
def print_diffs(rams):
    diff_states(rams[0], rams[1], 'start -> mart')
    diff_states(rams[1], rams[2], 'mart -> got_parcel')
    diff_states(rams[2], rams[3], 'got_parcel -> parcel_delivered')

def diff_states(ram_a, ram_b, message):
    print(message)

    workBlockSize = 4096
    workStartBlock = workRamStartHex - ramStartHex
    workEndBlock = workStartBlock + workBlockSize
    for i in range(workStartBlock, workEndBlock):
        if ram_a[i] != ram_b[i]:
            address_i = workRamStartHex + (i - workStartBlock)
            print(f'address: {hex(address_i)}, old_value: {int(ram_a[i])}, new_value: {int(ram_b[i])}')
            

def check_start_ram(info, ram):
    ramStartDec = 53248

    party_count_block = workRamStartHex - ramStartHex
    party_count_offset = 53602 - ramStartDec
    party_count_ram_index = party_count_block + party_count_offset
    print(party_count_ram_index)
    print('ram party count: ', int(ram[party_count_ram_index]))
    print('info party count: ', info['party_count'])


if __name__ == '__main__':
    run_experiments()