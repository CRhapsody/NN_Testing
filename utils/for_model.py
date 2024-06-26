
from collections import OrderedDict
def check_state(checkpoint):
    state = checkpoint['net']
    new_state_dict = OrderedDict()
    for key, value in state.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict