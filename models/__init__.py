"""Benchmark architectures package.

Two ldct-benchmark trunks (github.com/eeulig/ldct-benchmark,
commit 09b1011bc2fb77ef4fc734cec1e961a20c754910), one file per model:

  models/redcnn.py   RED-CNN (Chen et al. 2017)
  models/resnet.py   ResNet  (Park et al. 2017)

The study design locks RED-CNN + ResNet as the two mandatory trunks: every
arm (A..G) runs on both. Both map a 1-channel image to a 1-channel image,
so any plug-in output head can wrap either trunk.
"""

from models.redcnn import RedCNN
from models.resnet import ResNet

ARCHITECTURES = {
    "redcnn": RedCNN,
    "resnet": ResNet,
}

ARCH_CHOICES = tuple(ARCHITECTURES.keys())


def build_bare_model(name: str):
    """Instantiate an architecture by name (no device placement, no prints)."""
    key = name.lower().strip()
    if key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: '{name}'. "
            f"Use one of: {', '.join(ARCH_CHOICES)}."
        )
    return ARCHITECTURES[key]()


def build_benchmark_model(name: str, device):
    """Instantiate an architecture, move it to `device` and print a summary."""
    model = build_bare_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : {name.upper()}")
    print(f"  Parameters   : {n_params:,}")
    return model
