import sys
import os.path

import torch

from train import define_argparser
from train import main


def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for prev_key in vars(prev_config).keys():   
        if not prev_key in vars(config).keys():     # 예전 config가 없으면 알려줌
            # No such argument in current config. Ignore that value.
            print('WARNING!!! Argument "--%s" is not found in current argument parser.\tIgnore saved value:' % prev_key,
                  vars(prev_config)[prev_key])

    for key in vars(config).keys():
        if not key in vars(prev_config).keys():     # 예전에 없었는데 새로 생긴 config
            # No such argument in saved file. Use current value.
            print('WARNING!!! Argument "--%s" is not found in saved model.\tUse current value:' % key,
                  vars(config)[key])
        elif vars(config)[key] != vars(prev_config)[key]:   # 사용자가 이전 config 값을 새로 바꾸면 알려줌
            if '--%s' % key in sys.argv:                 # default값이 적용되지 않고 예전에 썼던 값이 들어가야 할 때
                # User changed argument value at this execution.
                print('WARNING!!! You changed value for argument "--%s".\tUse current value:' % key,
                      vars(config)[key])
            else:
                # User didn't changed at this execution, but current config and saved config is different.
                # This may caused by user's intension at last execution.
                # Load old value, and replace current value.
                vars(config)[key] = vars(prev_config)[key]      

    return config


def continue_main(config, main):
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.load_fn):          # config.load_fn으로 읽어옴
        saved_data = torch.load(config.load_fn, map_location='cpu') # map_location 지정 해줘야 함. 저장 될 때 하던 device로 자동으로 저장되는데 다른 작업을 다른 device로 하면 에러.

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)      # 김강사 노하우로 자랑거리..ㅎㅎㅎ

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']

        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Cannot find file %s' % config.load_fn)


if __name__ == '__main__':
    config = define_argparser(is_continue=True)  # train.py파일의 define_argparser(is_continue=False):였음
    continue_main(config, main)                  # True하면 그 밑의 애들이 바뀌어 작동함...ㅎㅎ --load fn
                                                 # --init epoch을 해줘야 함 전엔 1로 돼있어서 안했음, 지금은 11
                                                 # 이런 애들이 continue_main으로 가서 