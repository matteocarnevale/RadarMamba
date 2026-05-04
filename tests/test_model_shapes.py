"""
Test unitari sulle shape dei tensori lungo la pipeline del modello.
Verifica che ogni componente produca la shape attesa prima di
avere l'implementazione completa.

Esegui con: pytest tests/test_model_shapes.py -v

Molti test falliscono con NotImplementedError perché il codice
ha TODO — questo è intenzionale.
Usa il marker @pytest.mark.skip oppure implementa i moduli uno per volta.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import numpy as np

# Dimensioni ridotte per test veloci (non le dimensioni reali del paper)
B = 2   # batch size
R, A, E = 32, 48, 8    # griglia ridotta
D = 16   # Doppler bins


class TestFocalLoss:
    """FocalLoss è completamente implementata — questi test devono passare."""

    def test_output_scalar(self):
        from src.losses.focal_loss import FocalLoss
        loss_fn = FocalLoss(alpha=0.995, gamma=2.0)
        logits  = torch.randn(B, R, A, E)
        targets = torch.randint(0, 2, (B, R, A, E)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss deve essere uno scalare"

    def test_output_positive(self):
        from src.losses.focal_loss import FocalLoss
        loss_fn = FocalLoss()
        logits  = torch.randn(B, R, A, E)
        targets = torch.randint(0, 2, (B, R, A, E)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_sparse_target_high_alpha(self):
        """Con target sparsi (0.1% positivi) e α alto la loss non deve esplodere."""
        from src.losses.focal_loss import FocalLoss
        loss_fn = FocalLoss(alpha=0.995, gamma=2.0)
        logits  = torch.zeros(B, R, A, E)
        targets = torch.zeros(B, R, A, E)
        targets[0, 0, 0, 0] = 1.0   # solo un voxel positivo
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"Loss non finita: {loss}"


class TestMetrics:
    """Metriche sono completamente implementate — devono passare."""

    def test_chamfer_distance(self):
        from src.utils.metrics import chamfer_distance_numpy
        pred = np.random.randn(100, 3)
        gt   = np.random.randn(80, 3)
        cd   = chamfer_distance_numpy(pred, gt)
        assert cd > 0.0
        assert np.isfinite(cd)

    def test_chamfer_identical_clouds(self):
        from src.utils.metrics import chamfer_distance_numpy
        pts = np.random.randn(50, 3)
        cd  = chamfer_distance_numpy(pts, pts)
        assert cd < 1e-6, "CD di cloud identici deve essere ~0"

    def test_ucd_direction(self):
        """UCD è asimmetrica — diversa da CD bidirezionale."""
        from src.utils.metrics import chamfer_distance_numpy, unidirectional_chamfer_numpy
        pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gt   = np.array([[0.5, 0.0, 0.0]])
        cd  = chamfer_distance_numpy(pred, gt)
        ucd = unidirectional_chamfer_numpy(pred, gt)
        assert ucd <= cd   # UCD ≤ CD

    def test_metrics_empty(self):
        """Cloud vuoti → inf."""
        from src.utils.metrics import chamfer_distance_numpy
        cd = chamfer_distance_numpy(np.zeros((0, 3)), np.random.randn(10, 3))
        assert cd == float("inf")


class TestCFAR:
    """CFAR torch è implementato — deve passare."""

    def test_cfar_output_shape(self):
        from src.utils.cfar import ca_cfar_2d_torch
        power_map = torch.rand(64, 128)
        result = ca_cfar_2d_torch(power_map, guard_cells=2, reference_cells=4,
                                   threshold_factor=2.0, soft=True)
        assert result.shape == power_map.shape

    def test_cfar_output_range(self):
        """Output soft deve essere in [0, 1]."""
        from src.utils.cfar import ca_cfar_2d_torch
        power_map = torch.rand(32, 64) * 100.0
        result = ca_cfar_2d_torch(power_map, 2, 4, 2.0, soft=True)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestCBAM:
    """CBAM è completamente implementato — deve passare."""

    def test_output_shape(self):
        from src.models.cbam import CBAM
        cbam = CBAM(in_channels=64)
        x    = torch.randn(B, 64, R, A)
        out  = cbam(x)
        assert out.shape == x.shape, f"CBAM: attesa {x.shape}, ricevuta {out.shape}"

    def test_attention_modulation(self):
        """L'output non deve essere identico all'input (l'attenzione deve modulare)."""
        from src.models.cbam import CBAM
        cbam = CBAM(in_channels=16)
        x = torch.ones(1, 16, 8, 8)
        out = cbam(x)
        assert not torch.allclose(out, x), "CBAM deve modificare i valori"


class TestTemporalFusion:
    """TemporalFusion è implementato — deve passare."""

    def test_fuse_three_frames(self):
        from src.radar_tensor.temporal_fusion import fuse_three_frames
        t0 = np.random.rand(R, A, E, 2).astype(np.float32)
        t1 = np.random.rand(R, A, E, 2).astype(np.float32)
        t2 = np.random.rand(R, A, E, 2).astype(np.float32)
        fused = fuse_three_frames(t0, t1, t2)
        assert fused.shape == (R, A, E, 6)
        assert fused.dtype == np.float32

    def test_fuse_with_padding(self):
        """Con solo 1 frame, gli altri 2 devono essere zero-padded."""
        from src.radar_tensor.temporal_fusion import fuse_three_frames
        t0 = np.ones((R, A, E, 2), dtype=np.float32)
        fused = fuse_three_frames(t0, None, None, padding_mode="zeros")
        assert fused.shape == (R, A, E, 6)
        # Canali 2-5 devono essere zero
        np.testing.assert_array_equal(fused[..., 2:], 0.0)

    def test_temporal_buffer(self):
        """TemporalFusion buffer gestisce 3 frame correttamente."""
        from src.radar_tensor.temporal_fusion import TemporalFusion
        tf = TemporalFusion(n_frames=3, tensor_shape=(R, A, E))
        for _ in range(5):
            frame = np.random.rand(R, A, E, 2).astype(np.float32)
            fused = tf.push(frame)
            assert fused.shape == (R, A, E, 6)


# ── Test che richiedono implementazione (TODO) — skip di default ──────

@pytest.mark.skip(reason="Richiede implementazione di RMBlock (TODO)")
def test_rm_block_shape():
    from src.models.rm_block import RMBlock
    block = RMBlock(in_channels=64)
    x = torch.randn(B, 64, R, A)
    out = block(x)
    assert out.shape == x.shape


@pytest.mark.skip(reason="Richiede implementazione di RadarMambaUNet (TODO)")
def test_full_model_forward():
    from src.models.radar_mamba_unet import RadarMambaUNet
    model = RadarMambaUNet(n_elevation_bins=E)
    radar_cube = torch.randn(B, 6, R, A, E)
    rad_map    = torch.randn(B, D, R, A)
    logits = model(radar_cube, rad_map)
    assert logits.shape == (B, R, A, E), \
        f"Atteso (B,R,A,E)={(B,R,A,E)}, ricevuto {logits.shape}"
