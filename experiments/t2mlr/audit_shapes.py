"""Compare full checkpoint tensor metadata without downloading model weights."""
import json
import struct
import urllib.request
from pathlib import Path

import torch
from adapter import ExactAdapter, MODEL, REVISION


def main():
    root = Path(__file__).resolve().parents[2]
    snapshot = json.loads((root / "01_literature/t2mlr_public_artifacts_2026-09-18.json").read_text(encoding="utf-8-sig"))
    config = snapshot["config"]["result"]
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
    with urllib.request.urlopen(request, timeout=60) as response:
        count = struct.unpack("<Q", response.read(8))[0]
        header = json.loads(response.read(count))
    with torch.device("meta"):
        model = ExactAdapter(config, root / "external/T2MLR", dtype=torch.bfloat16)
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    actual = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    mismatch = {key: {"expected": expected.get(key), "actual": actual.get(key)}
                for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)}
    report = {"model": MODEL, "revision": REVISION, "tensor_count": len(actual),
              "shape_mismatches": mismatch, "weights_downloaded": False,
              "limitation": "Metadata parity only; tied tensor equality and BF16 inference remain unchecked"}
    (root / "04_evidence/t2mlr_shape_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
