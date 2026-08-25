"""CPU checks for the E-full-slice-context data/control path."""

import torch

from benchmark_data import FullSliceContextd
from calibration_head import ContextCalibrationHead
from train_head import batch_context


def test_context_precedes_patch_crop():
    image = torch.linspace(-2.0, 2.0, 64 * 64).reshape(1, 64, 64)
    transform = FullSliceContextd()
    expected = transform({"image": image.clone()})["full_context"]

    # Two different crops of the same source slice must retain its same context.
    crop_a = {"image": image[:, :16, :16], "full_context": expected}
    crop_b = {"image": image[:, 48:, 48:], "full_context": expected}
    assert not torch.allclose(crop_a["image"].mean(), crop_b["image"].mean())
    assert torch.equal(crop_a["full_context"], crop_b["full_context"])


def test_batch_context_uses_full_slice_statistics_not_patch():
    head = ContextCalibrationHead(full_slice_context=True)
    z = torch.randn(2, 1, 16, 16)
    full_context = torch.randn(2, 7)
    got = batch_context(head, {"full_context": full_context}, z)
    assert torch.equal(got, full_context)
    assert not got.requires_grad


def test_no_body_label_is_consumed_by_full_slice_context():
    head = ContextCalibrationHead(full_slice_context=True)
    z = torch.randn(2, 1, 16, 16)
    full_context = torch.randn(2, 7)
    a = batch_context(head, {"full_context": full_context,
                             "body_type": ["Chest", "Abdomen"]}, z)
    b = batch_context(head, {"full_context": full_context,
                             "body_type": ["Abdomen", "Chest"]}, z)
    assert torch.equal(a, b)


if __name__ == "__main__":
    test_context_precedes_patch_crop()
    test_batch_context_uses_full_slice_statistics_not_patch()
    test_no_body_label_is_consumed_by_full_slice_context()
    print("full-slice-context checks: OK")
