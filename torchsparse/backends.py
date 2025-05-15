import torch


def init():
    global benchmark, allow_tf32, allow_fp16, device_capability, hash_rsv_ratio
    benchmark = False
    device_capability = torch.xpu.get_device_capability()
    device_capability = device_capability.get('driver_version')
    allow_tf32 = device_capability >= 800
    allow_fp16 = device_capability.get('has_fp16')
    hash_rsv_ratio = 2  # default value, reserve 2x ( 2 * original_point_number) space for downsampling
