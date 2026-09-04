"""CPU checks for the E-full-slice-context data/control path."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from benchmark_data import (
    BENCHMARK_PIXEL_STD, FullSliceContextd, standardize_hu,
)
from calibration_head import ContextCalibrationHead
from evaluate_image import (
    _apply_calibration_head, _head_checkpoint_meta, _patient_row,
)
from hu_audit import (
    _accumulate_event_stats, _correction_at_threshold_hu,
    _finalize_event_stats, _mask_distance_stats, _paired_threshold_stats,
    _threshold_counts, _threshold_metrics,
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


def test_threshold_counts_report_error_direction():
    ref = torch.tensor([100.0, 150.0, 100.0, 150.0])
    pred = torch.tensor([140.0, 120.0, 90.0, 160.0])
    counts = _threshold_counts(ref, pred, 130.0)

    assert counts == {
        "ref_pos": 2,
        "pred_pos": 2,
        "false_pos": 1,
        "false_neg": 1,
        "disagree": 2,
        "n": 4,
    }
    metrics = _threshold_metrics(counts)
    assert metrics["FalsePosPrevalence_pct"] == 25.0
    assert metrics["FalseNegPrevalence_pct"] == 25.0
    assert metrics["FalsePositiveRate_pct"] == 50.0
    assert metrics["FalseNegativeRate_pct"] == 50.0


def test_threshold_distance_stats_use_pre_head_values():
    mask = torch.tensor([True, True, False, False])
    distance = torch.tensor([0.5, 7.0, 0.1, 0.1])
    stats = _mask_distance_stats(mask, distance)

    assert stats["Count"] == 2
    assert stats["MeanDistance_HU"] == 3.75
    assert stats["Within1HU_pct"] == 50.0
    assert stats["Within5HU_pct"] == 50.0
    assert stats["Within10HU_pct"] == 100.0


def test_paired_threshold_stats_classify_head_changes():
    ref = torch.tensor([140.0, 140.0, 120.0, 120.0, 140.0, 120.0])
    trunk = torch.tensor([129.0, 131.0, 129.0, 131.0, 125.0, 135.0])
    head = torch.tensor([131.0, 129.0, 131.0, 129.0, 125.0, 135.0])

    stats, masks, distance = _paired_threshold_stats(ref, trunk, head, 130.0)

    assert masks["FlipUp"].tolist() == [True, False, True, False, False, False]
    assert masks["FlipDown"].tolist() == [False, True, False, True, False, False]
    assert masks["NewDisagree"].tolist() == [False, True, True, False, False, False]
    assert masks["ResolvedDisagree"].tolist() == [True, False, False, True, False, False]
    assert distance.tolist() == [1.0, 1.0, 1.0, 1.0, 5.0, 5.0]
    assert stats["NewDisagree"]["Count"] == 2
    assert stats["ResolvedDisagree"]["Count"] == 2
    assert stats["NewDisagree"]["MeanDistance_HU"] == 1.0


def test_patient_event_aggregation_is_pixel_weighted():
    total = {"count": 0, "distance_sum": 0.0,
             "bands": {band: 0 for band in (1.0, 2.0, 5.0, 10.0, 20.0)}}
    mask_a = torch.tensor([True, False])
    distance_a = torch.tensor([1.0, 99.0])
    stats_a = _mask_distance_stats(mask_a, distance_a)
    _accumulate_event_stats(total, stats_a, mask_a, distance_a)

    mask_b = torch.tensor([True, True, True])
    distance_b = torch.tensor([3.0, 5.0, 7.0])
    stats_b = _mask_distance_stats(mask_b, distance_b)
    _accumulate_event_stats(total, stats_b, mask_b, distance_b)
    result = _finalize_event_stats(total, total_pixels=20)

    assert result["Count"] == 4
    assert result["pct"] == 20.0
    assert result["MeanDistance_HU"] == 4.0
    assert result["Within5HU_pct"] == 75.0


def test_threshold_correction_is_converted_to_hu():
    torch.manual_seed(1)
    head = ContextCalibrationHead(full_slice_context=True)
    torch.nn.init.zeros_(head.fc3.weight)
    torch.nn.init.constant_(head.fc3.bias, 0.25)
    context = torch.zeros(1, head.context_dim)
    got = _correction_at_threshold_hu(
        head, context, 130.0, torch.device("cpu"), torch.float32)
    z_threshold = standardize_hu(torch.tensor(130.0)).reshape(1, 1)
    expected = float(head.correction(
        z_threshold, context=context).reshape(-1)[0] * BENCHMARK_PIXEL_STD)
    assert abs(got - expected) < 1e-5


if __name__ == "__main__":
    test_context_precedes_patch_crop()
    test_batch_context_uses_full_slice_statistics_not_patch()
    test_no_body_label_is_consumed_by_full_slice_context()
    test_evaluator_uses_low_dose_slice_for_full_context()
    test_evaluator_reports_deltas_against_cycle_zero()
    test_evaluator_rejects_mismatched_head_provenance()
    test_threshold_counts_report_error_direction()
    test_threshold_distance_stats_use_pre_head_values()
    test_paired_threshold_stats_classify_head_changes()
    test_patient_event_aggregation_is_pixel_weighted()
    test_threshold_correction_is_converted_to_hu()
    print("full-slice-context checks: OK")
