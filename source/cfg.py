#   Developed by LIU Tianyi on 2019-08-20
#   Copyright(c) 2019. All Rights Reserved.

#   Train Val Test Parameters
params = {
    'lr': 1e-4,
    'lr_decay_epoch': 5,
    'lr_decay_rate': 1e-1,
    'batch_size': 256,
    'n_epoch': 30,
    'print_step': 5,
    'save_step': 5
}

#   Class Name
cls_name = [
    'B_Cells',
    'M_Cells',
    'T_Cells'
]

#   Class Range
cls_range = [
    'all',
    'all',
    2000
]

#   Class Color
cls_color = [
    'r',
    'g',
    'b'
]

#   Validate
def params_val():
    assert len(cls_name) == len(cls_range)
    assert len(cls_range) == len(cls_color)
    print("Parameters Validated")
    return None

