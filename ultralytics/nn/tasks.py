# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
import contextlib
import re
from copy import deepcopy
from pathlib import Path
import torch
from ultralytics.nn.modules import C2PSA, C3, SPPF, Bottleneck, C2f, C3k2, DetailSemanticFusion, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, RepConv, RepVGGDW
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts, model_info, time_sync

class BaseModel(torch.nn.Module):

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        (y, dt, embeddings) = ([], [], [])
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        LOGGER.warning(f"{self.__class__.__name__} does not support 'augment=True' prediction. Reverting to single-scale prediction.")
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        try:
            import thop
        except ImportError:
            thop = None
        c = m == self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1000000000.0 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, 'end2end', False):
                    m.fuse()
            self.info(verbose=verbose)
        return self

    def is_fused(self, thresh=10):
        bn = tuple((v for (k, v) in torch.nn.__dict__.items() if 'Norm' in k))
        return sum((isinstance(v, bn) for v in self.modules())) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        model = weights['model'] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        updated_csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(updated_csd, strict=False)
        len_updated_csd = len(updated_csd)
        first_conv = 'model.0.conv.weight'
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            (c1, c2, h, w) = state_dict[first_conv].shape
            (cc1, cc2, ch, cw) = csd[first_conv].shape
            if ch == h and cw == w:
                (c1, c2) = (min(c1, cc1), min(c2, cc2))
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f'Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch['img'])
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')

class DetectionModel(BaseModel):

    def __init__(self, cfg='yolo26n.yaml', ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.yaml['backbone'][0][2] == 'Silence':
            LOGGER.warning('YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. Please delete local *.pt file and re-download the latest model checkpoint.')
            self.yaml['backbone'][0][2] = 'nn.Identity'
        self.yaml['channels'] = ch
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        (self.model, self.save) = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                output = self.forward(x)
                if self.end2end:
                    output = output['one2many']
                return output['feats']
            self.model.eval()
            m.training = True
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    @property
    def end2end(self):
        return getattr(self.model[-1], 'end2end', False)

    @end2end.setter
    def end2end(self, value):
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        head = self.model[-1]
        for (k, v) in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head has no attribute '{k}'.")
                continue
            setattr(head, k, v)

    def _predict_augment(self, x):
        raise ValueError('The manuscript protocol uses augment=False')

    def init_criterion(self):
        return E2ELoss(self) if getattr(self, 'end2end', False) else v8DetectionLoss(self)

def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    (ckpt, weight) = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **ckpt.get('train_args', {})}
    model = (ckpt.get('ema') or ckpt['model']).float()
    model.args = args
    model.pt_path = str(weight)
    model.task = getattr(model, 'task', guess_model_task(model))
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.0])
    model = (model.fuse() if fuse and hasattr(model, 'fuse') else model).eval().to(device)
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and (not hasattr(m, 'recompute_scale_factor')):
            m.recompute_scale_factor = None
    return (model, ckpt)

def yaml_model_load(path):
    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub('(\\d+)([nslmx])6(.+)?$', '\\1\\2-p6\\3', path.stem)
        LOGGER.warning(f'Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)
    unified_path = re.sub('(\\d+)([nslmx])(.+)?$', '\\1\\3', str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d

def guess_model_scale(model_path):
    try:
        return re.search('yolo(e-)?[v]?\\d+([nslmx])', Path(model_path).stem).group(2)
    except AttributeError:
        return ''

def torch_safe_load(weight, safe_only=False):
    weight = Path(weight)
    if weight.suffix != '.pt' or not weight.is_file():
        raise FileNotFoundError('Supply a local published PyTorch checkpoint')
    ckpt = torch_load(weight, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError('Checkpoint must contain a model entry')
    return (ckpt, str(weight))

def guess_model_task(model):
    return 'detect'

def parse_model(d, ch, verbose=True):
    import ast
    legacy = True
    max_channels = float('inf')
    (nc, act, scales, end2end) = (d.get(x) for x in ('nc', 'activation', 'scales', 'end2end'))
    reg_max = d.get('reg_max', 16)
    (depth, width) = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple'))
    scale = d.get('scale')
    if scales:
        scale = scale or next(iter(scales))
        (depth, width, max_channels) = scales[scale]
    if act:
        Conv.default_act = eval(act)
    ch = [ch]
    (layers, save, c2) = ([], [], ch[-1])
    base_modules = {Conv, Conv2, DWConv, Bottleneck, C2f, C3, C3k2, C2PSA, SPPF}
    repeat_modules = {C2f, C3, C3k2, C2PSA}
    for (i, (f, n, module, args)) in enumerate(d['backbone'] + d['head']):
        m = getattr(torch.nn, module[3:]) if module.startswith('nn.') else globals()[module]
        for (j, a) in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in base_modules:
            (c1, c2) = (ch[f], args[0])
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in 'mlx':
                    args[3] = True
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum((ch[x] for x in f))
        elif m is DetailSemanticFusion:
            c2 = args[0] if args else ch[f[0]]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[x] for x in f], c2]
        elif m is Detect:
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            m.legacy = legacy
        else:
            c2 = ch[f]
        layer = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        layer.np = sum((x.numel() for x in layer.parameters()))
        (layer.i, layer.f, layer.type) = (i, f, str(m)[8:-2].replace('__main__.', ''))
        if verbose:
            LOGGER.info(f'{i:>3}{f!s:>20}{n_:>3}{layer.np:10.0f}  {layer.type:<45}{args!s:<30}')
        save.extend((x % i for x in ([f] if isinstance(f, int) else f) if x != -1))
        layers.append(layer)
        if i == 0:
            ch = []
        ch.append(c2)
    return (torch.nn.Sequential(*layers), sorted(save))
