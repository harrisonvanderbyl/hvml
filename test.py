

libloc = "CreateWindowWithCudaWrite.so"

from ctypes import CDLL, c_void_p, c_size_t
lib = CDLL(f"./{libloc}")

lib.init(800, 600)

import torch
atensor = torch.zeros((600, 800, 4), dtype=torch.uint8).cuda()
atensor[..., 0] = 255  # Red channel
atensor[..., 3] = 255  # Alpha channel

lib.copyDataToDisplay.argtypes = [c_void_p, c_size_t]


import time
current_time = time.time()
while True:
    # calc fps
    newcurrent_time = time.time()
    print("FPS:", 1.0 / (newcurrent_time - current_time))

    current_time = time.time()

    # create a torch tensor from the pointer
    lib.copyDataToDisplay(atensor.data_ptr(), atensor.numel()*atensor.element_size())
    lib.updateDisplay()