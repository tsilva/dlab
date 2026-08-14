from importlib.metadata import version

import onnx
import torch
from packaging.version import Version


def test_runtime_dependencies_are_outside_known_vulnerable_ranges() -> None:
    minimum_versions = {
        "gitpython": "3.1.58",
        "jupyterlab": "4.5.10",
        "onnx": "1.22.0",
        "pillow": "12.3.0",
        "setuptools": "83.0.0",
        "torch": "2.13.0",
        "torchvision": "0.28.0",
        "urllib3": "2.7.0",
    }

    for package, minimum in minimum_versions.items():
        assert Version(version(package)) >= Version(minimum)


def test_legitimate_torch_jit_and_onnx_models_remain_valid() -> None:
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    traced = torch.jit.trace(lambda value: value.relu() + 1, inputs)

    assert torch.equal(traced(inputs), inputs.relu() + 1)

    model = onnx.helper.make_model(onnx.helper.make_graph([], "smoke", [], []))
    onnx.checker.check_model(model)
