import math


def bit_plane_slicing(image):
    bit_plane_slices = []
    for k in range(8):
        new_slice = image % 2 * math.pow(2, k)
        image = image // 2
        bit_plane_slices.append(new_slice)
    return bit_plane_slices
