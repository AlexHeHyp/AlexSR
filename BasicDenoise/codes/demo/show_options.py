import os
from collections import OrderedDict
from datetime import datetime
import json

#--------------------------parse json file------------------------------#
def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt = dict_to_nonedict(opt)  #add by Alex He
    return opt


def save(dump_dir):
    dump_path = os.path.join(dump_dir, 'show_options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

#--------------------------parse json file------------------------------#
def get_window_show_option(show_opt):
    win_x = show_opt['windowsetting']['startwindow']['x']
    win_y = show_opt['windowsetting']['startwindow']['y']
    win_w = show_opt['windowsetting']['startwindow']['w']
    win_h = show_opt['windowsetting']['startwindow']['h']
    win_dist = show_opt['windowsetting']['distance']
    win_layout = show_opt['windowsetting']['layout']
    win_wait = show_opt['windowsetting']['waittime']

    return win_x, win_y, win_w, win_h, win_dist, win_layout, win_wait