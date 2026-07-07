import numpy as np
from geomex_vlm.model.spatial_tokens import encode_aab, decode_aab, encode_rab, decode_rab, encode_mask_rle, decode_mask_rle, rbox_to_aab


def test_aab_roundtrip():
    token = encode_aab((10, 20, 100, 120), (512, 512), resolution=1000)
    box = decode_aab(token, (512, 512), resolution=1000)
    assert abs(box[0] - 10) < 1.0
    assert abs(box[3] - 120) < 1.0


def test_rab_roundtrip_and_projection():
    token = encode_rab((50, 60, 30, 10, 45), (128, 128))
    rbox = decode_rab(token, (128, 128))
    aab = rbox_to_aab(rbox)
    assert len(aab) == 4
    assert aab[2] > aab[0]


def test_mask_rle_roundtrip():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    token = encode_mask_rle(mask)
    decoded = decode_mask_rle(token, (4, 4))
    assert np.array_equal(mask, decoded)
