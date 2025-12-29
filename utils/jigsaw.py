import torch
import random


def shuffle_blocks(feature_map, shuffled_order=None):
    # 获取特征图的形状
    B, C, H, W = feature_map.size()

    # 确保H和W可以被2整除
    assert H % 2 == 0 and W % 2 == 0, "Height and Width must be divisible by 2"

    # 分割特征图为四个块
    block1 = feature_map[:, :, :H // 2, :W // 2]
    block2 = feature_map[:, :, :H // 2, W // 2:]
    block3 = feature_map[:, :, H // 2:, :W // 2]
    block4 = feature_map[:, :, H // 2:, W // 2:]

    # 将四个块放入列表中
    blocks = [block1, block2, block3, block4]

    # 记录原始的块顺序
    original_order = [0, 1, 2, 3]

    # 随机打乱块顺序
    if shuffled_order is None:
        shuffled_order = original_order[:]
        random.shuffle(shuffled_order)
    else:
        shuffled_order = shuffled_order

    # 根据打乱后的顺序重新组合特征图
    shuffled_feature_map = torch.zeros_like(feature_map)
    for idx, block_idx in enumerate(shuffled_order):
        if idx == 0:
            shuffled_feature_map[:, :, :H // 2, :W // 2] = blocks[block_idx]
        elif idx == 1:
            shuffled_feature_map[:, :, :H // 2, W // 2:] = blocks[block_idx]
        elif idx == 2:
            shuffled_feature_map[:, :, H // 2:, :W // 2] = blocks[block_idx]
        else:
            shuffled_feature_map[:, :, H // 2:, W // 2:] = blocks[block_idx]

    # 返回打乱后的特征图和打乱顺序的标记
    return shuffled_feature_map, shuffled_order


def unshuffle_blocks(feature_map, shuffled_order):
    # 获取特征图的形状
    B, C, H, W = feature_map.size()

    # 确保H和W可以被2整除
    assert H % 2 == 0 and W % 2 == 0, "Height and Width must be divisible by 2"

    # 分割特征图为四个块
    block1 = feature_map[:, :, :H // 2, :W // 2]
    block2 = feature_map[:, :, :H // 2, W // 2:]
    block3 = feature_map[:, :, H // 2:, :W // 2]
    block4 = feature_map[:, :, H // 2:, W // 2:]

    # 将四个块放入列表中，并根据标记恢复顺序
    original_order = [0, 1, 2, 3]
    blocks = [None] * 4
    for idx, block_idx in enumerate(shuffled_order):
        blocks[block_idx] = [block1, block2, block3, block4][idx]

    # 根据原始顺序重新组合特征图
    unshuffled_feature_map = torch.zeros_like(feature_map)
    for idx in original_order:
        block = blocks[idx]
        if idx == 0:
            unshuffled_feature_map[:, :, :H // 2, :W // 2] = block
        elif idx == 1:
            unshuffled_feature_map[:, :, :H // 2, W // 2:] = block
        elif idx == 2:
            unshuffled_feature_map[:, :, H // 2:, :W // 2] = block
        else:
            unshuffled_feature_map[:, :, H // 2:, W // 2:] = block

    return unshuffled_feature_map