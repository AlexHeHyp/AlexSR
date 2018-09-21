global _global_dict
global g_show_train_img
global g_show_val_img
_global_dict = {}
g_show_train_img = False
g_show_val_img = False


def set_value(key, value):
    _global_dict[key] = value

def get_value(key, defvalue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defvalue


def set_t_value(val=True):
    global g_show_train_img
    g_show_train_img = val
    print("set_t_value:", g_show_train_img)

def get_t_value():
    global g_show_train_img
    return g_show_train_img


def set_v_value(val=True):
    global g_show_val_img
    g_show_val_img = val
    print("set_v_value:", g_show_val_img)

def get_v_value():
    global g_show_val_img
    return g_show_val_img
