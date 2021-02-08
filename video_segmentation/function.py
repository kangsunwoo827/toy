import cv2
import numpy as np

#part_ids = [ 0,  2,  4,  5,  6,  8, 13, 14, 16, 17, 18, 19, 20, 21, 24]
# the order is: (left right flipped)
# background, head, torso, left upper arm ,right upper arm, left forearm, right forearm,
#  left hand, right hand, left thigh, right thigh, left shank, right shank, left foot, right foot
# part_ids = [0, 13, 2, 5, 8, 19, 20, 4, 24, 18, 6, 21, 16, 14, 17]


# r_chan = [0, 127, 255, 255, 255, 127, 255, 127, 0, 0, 0, 0, 127, 255, 255]
# g_chan = [0, 127, 0, 127, 255, 0, 0, 127, 255, 0, 255, 127, 255, 127, 255]
# b_chan = [0, 127, 0, 0, 0, 255, 255, 0, 255, 255, 0, 255, 127, 127, 127]

index0_rgb=[255, 255, 255] #bg
index1_rgb=[127, 127, 127] #head
index2_rgb=[255, 0, 0] # torso
index3_rgb=[0, 255, 0] # upperarm
index4_rgb=[0, 255, 0] # forearm
index5_rgb=[0, 0, 255] # thigh
index6_rgb=[0, 0, 255] # calf


rgb_zip=list(zip(index0_rgb, index1_rgb, index2_rgb, index3_rgb, index4_rgb, index5_rgb, index6_rgb))

r_chan = rgb_zip[0]
g_chan = rgb_zip[1]
b_chan = rgb_zip[2]


png_idx_7parts = [0, 1, 2, 3, 4, 5, 6]
'''
def human_seg_spread_channel(human_seg_map):
    x = human_seg_map // 127
    x = x * np.array([9, 3, 1])
    x = np.add.reduce(x, 2)
    res = []
    for i in part_ids:
        res.append((x == i))
    res = np.stack(res, axis=-1)
    return res.astype(np.float32)
'''

def human_seg_combine_channel(human_seg_split_map):
    segmap = np.add.reduce(human_seg_split_map * np.array(png_idx_7parts), 2)
    return np.stack([segmap], axis=-1).astype(np.uint8)

def human_seg_combine_channel_rgb(human_seg_split_map):
    r_chan_seg = np.add.reduce(human_seg_split_map * np.array(r_chan), 2)
    g_chan_seg = np.add.reduce(human_seg_split_map * np.array(g_chan), 2)
    b_chan_seg = np.add.reduce(human_seg_split_map * np.array(b_chan), 2)
    return np.stack([b_chan_seg, g_chan_seg, r_chan_seg], axis=-1).astype(np.uint8)

def human_seg_combine_argmax(human_seg_argmax_map):
    onehot = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in range(7)], axis=-1)
    return human_seg_combine_channel(onehot)


def human_seg_combine_argmax_rgb(human_seg_argmax_map):
    onehot = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in range(7)], axis=-1)
    return human_seg_combine_channel_rgb(onehot)