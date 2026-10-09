"""Load a prompt once and render observations without filesystem access."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from settings import SceneConfig

PROMPT_PATH = Path(__file__).resolve().parent / 'prompt' / 'prompt.txt'
PROMPT_SEPARATOR = '\n--- EASG OBSERVATION TEMPLATE ---\n'


def render_config_text(text, config):
    return text.replace('[GRIPPER_MAX_CM]', f'{config.gripper_max_width_m * 100:g}')


def split_prompt_template(text):
    """Optional fixed system policy followed by the per-observation template."""
    if PROMPT_SEPARATOR not in text:
        return None, text
    if text.count(PROMPT_SEPARATOR) != 1:
        raise ValueError('Prompt must contain exactly one observation separator')
    policy, _, observation = text.partition(PROMPT_SEPARATOR)
    if not policy.strip() or not observation.strip():
        raise ValueError('Prompt policy and observation template must both be nonempty')
    return policy, observation


@dataclass(frozen=True)
class PromptTemplate:
    source: str
    system: str | None
    observation: str
    config: SceneConfig

    def render(self, info, instruction, previous):
        config = self.config
        clearance = info['clearance_m']
        depth_info = ('Unavailable: no valid visible surface was found near the TCP +Z ray. '
                      'No blue surface endpoint is drawn.' if clearance is None else
                      f'Visible surface clearance along TCP +Z: {clearance * 100:.2f} cm. '
                      'The blue endpoint is the selected RGB-D surface sample, not the image edge.')
        gripper = float(np.clip(info['gripper_width_m'] / config.gripper_max_width_m, 0., 1.))
        task = (instruction + '\n\nPrevious execution feedback:\n' + previous +
                f'\n\nSimulator hard caps (not suggested step sizes): translation at most {config.max_translation_cm:g} cm; absolute rotation at most {config.max_rotation_deg:g} degrees. Follow any stricter limits in the template. '
                'Opening executes before translation; closing executes after translation. '
                'Release using a separate zero-motion open command.')
        values = {'[TCP_Z_DISTANCE_INFO]': depth_info, '[Gripper state]': f'{gripper:+.2f}', '[Task]': task}
        prompt = self.observation
        for key, value in values.items():
            prompt = prompt.replace(key, value)
        geometry_keys = ('[CAMERA_INTRINSICS]', '[TCP_PROJECTION]', '[TCP_HEIGHT]', '[TCP_ALIGNMENT]', '[CAMERA_TCP_OFFSET]')
        if '[ROBOT_GEOMETRY]' in prompt or any(key in prompt for key in geometry_keys):
            uv = info['tcp_projected_pixels']
            endpoint = ('unavailable' if len(uv) < 4 else f'({uv[3][0]:.1f}, {uv[3][1]:.1f})')
            k = info['wrist_K']
            down = np.dot(np.asarray(info['tcp_world'])[:3, 2], [0., 0., -1.])
            size = info.get('wrist_resolution', [640, 480])
            offset = np.asarray(info.get('tcp_in_wrist_camera_m', [0., .08, .065])) * 100
            axes = np.asarray(info.get('tcp_axes_in_wrist_camera', np.eye(3)))
            same_axes = np.allclose(axes, np.eye(3), atol=1e-3)
            fields = {
                '[CAMERA_INTRINSICS]': (f'Wrist image: {size[0]} x {size[1]} pixels; origin top-left; u right, v down. '
                    f'fx={k[0][0]:.2f}, fy={k[1][1]:.2f}, cx={k[0][2]:.2f}, cy={k[1][2]:.2f}.'),
                '[TCP_PROJECTION]': f'TCP cross pixel=({uv[0][0]:.1f}, {uv[0][1]:.1f}); BLUE surface endpoint pixel={endpoint}.',
                '[TCP_HEIGHT]': f'Current TCP height above the support workmat H={info["tcp_height_above_workmat_m"]*100:.2f} cm.',
                '[TCP_ALIGNMENT]': f'Gripper +Z alignment with vertically down={down:.4f} (1 means down).',
                '[CAMERA_TCP_OFFSET]': (f'Camera-to-TCP offset in camera coordinates is ({offset[0]:+.2f}, {offset[1]:+.2f}, {offset[2]:+.2f}) cm. '
                    f'Camera axes equal TCP axes: {bool(same_axes)}. These are sensor/robot measurements, not object coordinates.')}
            geometry = (fields['[CAMERA_INTRINSICS]'] + '\n' + fields['[TCP_PROJECTION]'] + '\n' +
                        fields['[TCP_HEIGHT]'] + ' ' + fields['[TCP_ALIGNMENT]'] + '\n' + fields['[CAMERA_TCP_OFFSET]'])
            prompt = prompt.replace('[ROBOT_GEOMETRY]', geometry)
            for key, value in fields.items():
                prompt = prompt.replace(key, value)
        prompt = prompt.replace('[OBSERVATION_ID]', str(info.get('observation_id', 'LATEST')))
        return prompt


def load_prompt(path, config, *, system_path=None):
    source = Path(path).read_text()
    system, observation = split_prompt_template(render_config_text(source, config))
    for token in ('[Task]', '[Gripper state]', '[TCP_Z_DISTANCE_INFO]'):
        if token not in observation:
            raise ValueError(f'Paper prompt is missing placeholder {token}')
    if system_path is not None:
        system = render_config_text(Path(system_path).read_text(), config)
    return PromptTemplate(source, system, observation, config)
