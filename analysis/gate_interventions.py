"""Run the paper's frozen, padding-preserving DSF interventions; never trains."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime"))
from analysis.diagnostics import transform_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True, help="COCO JSON containing original image IDs, names and dimensions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-constant", action="store_true", help="Add G=0/G=1 validation controls")
    parser.add_argument("--include-full-canvas", action="store_true", help="Add the DSF+refinement validation controls described in Appendix A")
    args = parser.parse_args()
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import ops
    torch.set_num_threads(2)
    model_wrapper = YOLO(str(args.weights))
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    model = model_wrapper.model.float().eval().fuse(verbose=False).to(device).requires_grad_(False)
    dsf_index = next(i for i, module in enumerate(model.model) if type(module).__name__ == "DetailSemanticFusion")
    dsf, refinement, head = model.model[dsf_index], model.model[dsf_index + 1], model.model[-1]
    if args.include_constant or args.include_full_canvas:
        assert dsf.detail_proj.conv.out_channels == 32, "Constant/full-canvas controls belong to the DSF+refinement validation study"
    assert head.end2end and head.f == [dsf_index + 1, 22, 25, 28]
    head.max_det, head.agnostic_nms = 300, False
    cache = {}
    handles = [dsf.register_forward_pre_hook(lambda module, inputs: cache.update(inputs=inputs[0]))]
    for name in ("detail_proj", "semantic_mix", "gate"):
        handles.append(getattr(dsf, name).register_forward_hook(lambda module, inputs, out, key=name: cache.update({key: out})))
    handles.append(head.register_forward_hook(lambda module, inputs, out: cache.update(head_output=out)))
    images = json.loads(args.image_map.read_text(encoding="utf-8-sig"))["images"]
    if args.limit:
        images = images[:args.limit]
    by_name = {Path(im["file_name"]).name: im for im in images}
    cy, cx = torch.meshgrid(torch.arange(208) + .5, torch.arange(208) + .5, indexing="ij")
    modes = ["native", "valid_spatial_mean", "valid_shuffle_0", "valid_shuffle_l1_0"]
    if args.include_constant:
        modes += ["zero", "one"]
    if args.include_full_canvas:
        modes += ["spatial_mean", "shuffle_0", "shuffle_1", "shuffle_2", "shuffle_l1_0"]
    outputs = {mode: [] for mode in modes}
    checks = []

    def decode(out, im):
        out = out[0] if isinstance(out, tuple) else out
        rows = out[0][out[0, :, 4] > .001][:300].clone()
        rows[:, :4] = ops.scale_boxes((832, 832), rows[:, :4], (im["height"], im["width"]))
        return [dict(image_id=im["id"], category_id=int(c), score=float(score), bbox=[float(x), float(y), float(x2-x), float(y2-y)])
                for x, y, x2, y2, score, c in rows.cpu().numpy()]

    try:
        with torch.inference_mode():
            stream = model_wrapper.predict(source=str(args.images), stream=True, imgsz=832, batch=1, rect=False,
                conf=.001, iou=.7, max_det=300, half=False, device=args.device, augment=False,
                agnostic_nms=False, save=False, verbose=False)
            for result in stream:
                name = Path(result.path).name
                if name not in by_name:
                    continue
                im = by_name[name]
                assert result.orig_shape == (im["height"], im["width"])
                d, s, g = (cache[k] for k in ("detail_proj", "semantic_mix", "gate"))
                assert d.shape == s.shape == g.shape and tuple(g.shape[-2:]) == (208, 208)
                native = cache["head_output"]
                native = native[0] if isinstance(native, tuple) else native
                native_predictions = decode(native, im)
                actual = np.array([[p["bbox"][0], p["bbox"][1], p["bbox"][0]+p["bbox"][2],
                    p["bbox"][1]+p["bbox"][3], p["score"], p["category_id"]] for p in native_predictions],dtype=np.float32).reshape(-1,6)
                assert actual.shape == tuple(result.boxes.data.shape)
                assert np.allclose(actual,result.boxes.data.cpu().numpy(),atol=1e-4,rtol=0)
                outputs["native"].extend(native_predictions)
                replay = head([refinement(dsf.out(d + g*s)), *cache["inputs"][1:]])
                replay = replay[0] if isinstance(replay, tuple) else replay
                replay_error = float((native - replay).abs().max())
                assert replay_error <= 1e-5
                _, t = transform_boxes([[0, 0, im["width"], im["height"]]], im)
                mask = ((cx >= t["left"]/4) & (cx < (t["left"]+t["resized_width"])/4)
                        & (cy >= t["top"]/4) & (cy < (t["top"]+t["resized_height"])/4))
                cpu_ix = torch.nonzero(mask.flatten(), as_tuple=True)[0]
                draw = torch.randperm(len(cpu_ix), generator=torch.Generator().manual_seed(1000003 * im["id"]))
                ix, permutation = cpu_ix.to(g.device), cpu_ix[draw].to(g.device)
                padding = (~mask.flatten()).to(g.device)
                native_injection = g*s
                relative_l1_error = None
                for mode in modes[1:]:
                    gg = g.clone().flatten(2)
                    if mode == "zero":
                        gg.zero_()
                    elif mode == "one":
                        gg.fill_(1)
                    elif mode == "spatial_mean":
                        gg[:] = g.flatten(2).mean(2,keepdim=True)
                    elif mode.startswith("shuffle_"):
                        repeat=0 if mode=="shuffle_l1_0" else int(mode[-1])
                        full_perm=torch.randperm(208*208,generator=torch.Generator().manual_seed(1000003*im["id"]+repeat)).to(g.device)
                        gg[:]=g.flatten(2)[:,:,full_perm]
                    else:
                        gg[:, :, ix] = (g.flatten(2)[:, :, ix].mean(2, keepdim=True) if mode == "valid_spatial_mean"
                                        else g.flatten(2)[:, :, permutation])
                    injection = gg.reshape_as(g)*s
                    if mode in ("valid_shuffle_l1_0","shuffle_l1_0"):
                        selected=ix if mode=="valid_shuffle_l1_0" else torch.arange(208*208,device=g.device)
                        target = native_injection.flatten(2)[:, :, selected].abs().sum()
                        scale = target / injection.flatten(2)[:, :, selected].abs().sum().clamp_min(1e-12)
                        injection.flatten(2)[:, :, selected] *= scale
                        actual_l1 = injection.flatten(2)[:, :, selected].abs().sum()
                        relative_l1_error = float((actual_l1-target).abs()/target.clamp_min(1e-12))
                        assert relative_l1_error < 1e-5
                    if mode.startswith("valid_"):
                        assert torch.equal(gg[:, :, padding], g.flatten(2)[:, :, padding])
                        assert torch.equal(injection.flatten(2)[:, :, padding], native_injection.flatten(2)[:, :, padding])
                    outputs[mode].extend(decode(head([refinement(dsf.out(d+injection)), *cache["inputs"][1:]]), im))
                checks.append(dict(image_id=im["id"], replay_max_abs_difference=replay_error,
                    valid_cells=len(cpu_ix), l1_relative_error=relative_l1_error))
                print(f"Interventions {len(checks)}/{len(images)}", flush=True)
                if len(checks) == len(images):
                    break
    finally:
        for handle in handles:
            handle.remove()
    assert len(checks) == len(images), "Missing image(s) in supplied image directory"
    args.output.mkdir(parents=True, exist_ok=True)
    for mode, predictions in outputs.items():
        (args.output/f"predictions_{mode}.json").write_text(json.dumps(predictions, allow_nan=False), encoding="utf-8")
    with (args.output/"intervention_checks.csv").open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    report=dict(training=False, frozen_parameters=True, images=len(checks), modes=modes,
        weight_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(), input=832, precision="FP32",
        shuffle_rule="CPU generator seed=1000003*image_id, shared permutation across channels",
        l1_matching="Injected residual over valid cells; one scalar; padding preserved",
        constant_controls="G=0/G=1 modify whole canvas, as in validation control protocol")
    (args.output/"protocol.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
