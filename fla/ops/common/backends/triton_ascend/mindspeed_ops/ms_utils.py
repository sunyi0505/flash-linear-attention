# Migrated from MindSpeed-Ops mindspeed_ops/api/triton/utils.py (get_autotune_config)
from __future__ import annotations

import itertools

import triton


def get_autotune_config(
    multibuffer_list: tuple = (False,),
    unit_flag_list: tuple = (False,),
    limit_auto_multi_buffer_only_for_local_buffer_list: tuple = (False,),
    limit_auto_multi_buffer_of_local_buffer_list: tuple = ("no-l0c",),
    set_workspace_multibuffer_list: tuple = (2, 4),
    enable_hivm_auto_cv_balance_list: tuple = (True,),
    tile_mix_vector_loop_num_list: tuple = (2, 4),
    tile_mix_cube_loop_num_list: tuple = (2, 4),
):
    configs = []
    for (
        multibuffer,
        unit_flag,
        limit_auto_multi_buffer_only_for_local_buffer,
        limit_auto_multi_buffer_of_local_buffer,
    ) in itertools.product(
        list(multibuffer_list),
        list(unit_flag_list),
        list(limit_auto_multi_buffer_only_for_local_buffer_list),
        list(limit_auto_multi_buffer_of_local_buffer_list),
    ):
        base_config_dict = {
            'multibuffer': multibuffer,
            'unit_flag': unit_flag,
            'limit_auto_multi_buffer_only_for_local_buffer': limit_auto_multi_buffer_only_for_local_buffer,
            'limit_auto_multi_buffer_of_local_buffer': limit_auto_multi_buffer_of_local_buffer,
        }

        if limit_auto_multi_buffer_only_for_local_buffer:
            configs.append(triton.Config(base_config_dict))
        else:
            for (
                set_workspace_multibuffer,
                enable_hivm_auto_cv_balance,
                tile_mix_vector_loop,
                tile_mix_cube_loop,
            ) in itertools.product(
                list(set_workspace_multibuffer_list),
                list(enable_hivm_auto_cv_balance_list),
                list(tile_mix_vector_loop_num_list),
                list(tile_mix_cube_loop_num_list),
            ):
                config_dict = {
                    **base_config_dict,
                    'set_workspace_multibuffer': set_workspace_multibuffer,
                    'enable_hivm_auto_cv_balance': enable_hivm_auto_cv_balance,
                    'tile_mix_vector_loop': tile_mix_vector_loop,
                    'tile_mix_cube_loop': tile_mix_cube_loop,
                }
                configs.append(triton.Config(config_dict))
    return configs
