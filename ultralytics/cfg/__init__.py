# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from ultralytics.utils import DEFAULT_CFG_DICT, FLOAT_OR_INT, LOGGER, RANK, ROOT, RUNS_DIR, STR_OR_PATH, TESTS_RUNNING, YAML, IterableSimpleNamespace, colorstr, deprecation_warn
CFG_FLOAT_KEYS = frozenset({'warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear', 'time', 'workspace', 'batch', 'dn_geo_weight', 'dn_sem_weight', 'dn_temperature', 'p2_loss_gain', 'p3_loss_gain'})
CFG_FRACTION_KEYS = frozenset({'dropout', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'conf', 'iou', 'fraction', 'multi_scale'})
CFG_INT_KEYS = frozenset({'epochs', 'patience', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride', 'line_width', 'nbs', 'save_period'})
CFG_BOOL_KEYS = frozenset({'save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'dn_mask_same_source', 'val', 'save_json', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop', 'save_frames', 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks', 'show_boxes', 'keras', 'optimize', 'int8', 'dynamic', 'simplify', 'nms', 'profile', 'end2end'})

def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg

def get_cfg(cfg: str | Path | dict | SimpleNamespace=DEFAULT_CFG_DICT, overrides: dict | None=None) -> SimpleNamespace:
    cfg = cfg2dict(cfg)
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}
    for k in ('project', 'name'):
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':
        cfg['name'] = str(cfg.get('model', '')).partition('.')[0]
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")
    check_cfg(cfg)
    return IterableSimpleNamespace(**cfg)

def check_cfg(cfg: dict, hard: bool=True) -> None:
    for (k, v) in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and (not isinstance(v, FLOAT_OR_INT)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                    cfg[k] = v = float(v)
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and (not isinstance(v, int)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')")
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and (not isinstance(v, bool)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")
                cfg[k] = bool(v)

def get_save_dir(args: SimpleNamespace, name: str | None=None) -> Path:
    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path
        project = args.project or ''
        if not Path(project).is_absolute():
            project = (ROOT.parent / 'tests/tmp/runs' if TESTS_RUNNING else RUNS_DIR) / args.task / project
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)
    return Path(save_dir).resolve()

def _handle_deprecation(custom: dict) -> dict:
    deprecated_mappings = {'boxes': ('show_boxes', lambda v: v), 'hide_labels': ('show_labels', lambda v: not bool(v)), 'hide_conf': ('show_conf', lambda v: not bool(v)), 'line_thickness': ('line_width', lambda v: v)}
    removed_keys = {'label_smoothing', 'save_hybrid', 'crop_fraction'}
    for (old_key, (new_key, transform)) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))
    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)
    return custom

def check_dict_alignment(base: dict, custom: dict, e: Exception | None=None, allowed_custom_keys: set | None=None) -> None:
    custom = _handle_deprecation(custom)
    (base_keys, custom_keys) = (frozenset(x.keys()) for x in (base, custom))
    if allowed_custom_keys is None:
        allowed_custom_keys = {'augmentations', 'save_dir'}
    if (mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]):
        from difflib import get_close_matches
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f'{k}={base[k]}' if base.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e
if __name__ == '__main__':
    entrypoint(debug='')
TASKS = frozenset({'detect'})
MODES = frozenset({'train', 'val', 'predict'})
TASK2DATA = {'detect': 'configs/data/visdrone.example.yaml'}
CLI_HELP_MSG = 'Use the manuscript reproduction entry points in scripts/.'
