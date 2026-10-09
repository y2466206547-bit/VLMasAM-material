"""Physical/task settings loaded by name from EASG/config, independent of cwd."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'


@dataclass(frozen=True)
class SceneConfig:
    gripper_max_width_m: float
    max_translation_cm: float
    max_rotation_deg: float
    table_height_m: float
    workspace_x_m: list[float]
    workspace_y_m: list[float]
    workspace_z_above_table_m: list[float]
    action_duration_s: float
    arm_duration_s: float
    gripper_duration_s: float
    initial_pose_duration_s: float
    action_settle_s: float
    completion_settle_s: float
    force_backoff_duration_s: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.startswith('workspace_'):
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f'{field.name} must be [min, max]')
                if not all(type(v) in (int, float) and math.isfinite(v) for v in value) or value[0] >= value[1]:
                    raise ValueError(f'{field.name} needs finite min < max')
            elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name} must be positive and finite')
        # This is an asset limit, not a configurable opening target.
        if self.gripper_max_width_m > 0.08:
            raise ValueError('gripper_max_width_m exceeds the Franka asset limit of 0.08 m')
        if self.workspace_z_above_table_m[0] <= 0.004:
            raise ValueError('workspace Z minimum must clear the 0.004 m workmat')

    @property
    def workmat_top_m(self):
        return self.table_height_m + 0.004

    def contains_tcp(self, position):
        x, y, z = position
        low, high = self.workspace_z_above_table_m
        return (self.workspace_x_m[0] < x < self.workspace_x_m[1]
                and self.workspace_y_m[0] < y < self.workspace_y_m[1]
                and self.table_height_m + low <= z <= self.table_height_m + high)

    def to_dict(self):
        return asdict(self)


def load_config(name='default.json'):
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('--config must be a filename in EASG/config, not a path')
    path = CONFIG_DIR / name
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a JSON object')
    expected = {field.name for field in fields(SceneConfig)}
    if set(data) != expected:
        raise ValueError(f'Invalid config fields: missing={sorted(expected-set(data))}, unknown={sorted(set(data)-expected)}')
    return SceneConfig(**data)


def config_from_args(parser, args):
    try:
        return load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
