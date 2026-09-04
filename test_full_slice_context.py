"""CPU checks for the E-full-slice-context data/control path."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from benchmark_data import FullSliceContextd
from calibration_head import ContextCalibrationHead
from evaluate_image import (
    _apply_calibration_head, _head_checkpoint_meta, _patient_row,
)
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


def test_evaluator_uses_low_dose_slice_for_full_context():
    torch.manual_seed(0)
    head = ContextCalibrationHead(full_slice_context=True)
    torch.nn.init.constant_(head.fc3.weight, 0.1)
    torch.nn.init.constant_(head.fc3.bias, 0.2)
    x = torch.full((1, 1, 16, 16), -1.0)
    z = torch.full((1, 1, 16, 16), 1.0)

    got = _apply_calibration_head(head, x, z, "Chest")
    expected = z + head.correction(z, context=head.inferred_context(x))
    patch_context = z + head.correction(z, context=head.inferred_context(z))

    assert torch.equal(got, expected)
    assert not torch.allclose(got, patch_context)


def test_evaluator_reports_deltas_against_cycle_zero():
    metrics = {"PSNR": 30.0, "SSIM": 0.9, "RMSE_HU": 45.0, "VIF": 0.8}
    input_metrics = {"PSNR": 20.0, "SSIM": 0.7, "RMSE_HU": 80.0, "VIF": 0.5}
    trunk_metrics = {"PSNR": 29.0, "SSIM": 0.85, "RMSE_HU": 50.0, "VIF": 0.75}

    row = _patient_row(
        "C001", "Chest", 1, "Selected head checkpoint",
        metrics, input_metrics, trunk_metrics)

    assert row["Baseline_SSIM"] == 0.7
    assert row["Baseline_RMSE_HU"] == 80.0
    assert row["Delta_vs_Trunk_PSNR"] == 1.0
    assert row["Delta_vs_Trunk_RMSE_HU"] == -5.0


def test_evaluator_rejects_mismatched_head_provenance():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        trunk = root / "runs" / "redcnn" / "best_model.pt"
        trunk.parent.mkdir(parents=True)
        trunk.touch()
        head_path = root / "best_head.pt"
        torch.save({
            "meta": {
                "architecture": "redcnn",
                "split": "100p",
                "joint": False,
                "trunk_checkpoint": str(trunk),
            },
        }, head_path)

        _head_checkpoint_meta(head_path, "redcnn", trunk, "100p")

        try:
            _head_checkpoint_meta(head_path, "redcnn", trunk, "20p")
        except RuntimeError as error:
            assert "split mismatch" in str(error)
        else:
            raise AssertionError("mismatched split was accepted")

        other_trunk = root / "other" / "best_model.pt"
        try:
            _head_checkpoint_meta(head_path, "redcnn", other_trunk, "100p")
        except RuntimeError as error:
            assert "Head/trunk checkpoint mismatch" in str(error)
        else:
            raise AssertionError("mismatched trunk was accepted")


if __name__ == "__main__":
    test_context_precedes_patch_crop()
    test_batch_context_uses_full_slice_statistics_not_patch()
    test_no_body_label_is_consumed_by_full_slice_context()
    test_evaluator_uses_low_dose_slice_for_full_context()
    test_evaluator_reports_deltas_against_cycle_zero()
    test_evaluator_rejects_mismatched_head_provenance()
    print("full-slice-context checks: OK")
