import math
from typing import Optional

import numpy as np


def fractional_repeat(arr, repeat, target_length: Optional[int] = None):

    result = []
    carry = 0
    for elt in arr:
        floor_repeat = math.floor(repeat)
        curr_carry = repeat - floor_repeat
        carry += curr_carry

        result += [elt] * floor_repeat

        if carry >= 1:
            floor_carry = math.floor(carry)
            result += [elt] * floor_carry
            carry = carry - floor_carry

    if target_length is not None:
        result = result[:target_length]

    return np.array(result)


if __name__ == "__main__":
    array = np.array([0, 1, 2, 3, 4])
    repeated = fractional_repeat(array, 2.3)
    print(repeated)
