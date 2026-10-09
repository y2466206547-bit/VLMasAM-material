"""Closed-loop VLMasAM on EASG. No scripted grasp trajectory or object-pose oracle."""
import json
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from construct_scene.init_scene import PROJECT_ROOT, arguments, start_app
from settings import config_from_args
from client import (CHAT_CONNECTION_MODES, VisionClient, DEFAULT_REASONING_EFFORT,
                    DEFAULT_IMAGE_ENCODING, default_max_tokens)
from parse_util import parse_vlm_motion, motion_to_dict
from motion import TCP_FRAME, build_target_pose, validate_motion_limits, paper_to_hand_motion
from observation import capture
from process_prompt import PROMPT_PATH, load_prompt
from construct_scene.manipulation_scene import ManipulationScene, TCP_OFFSET


def prompt_for(scene, info, instruction, previous):
    return scene.prompt_template.render(info, instruction, previous)


def execute(scene, motion):
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    config = scene.config
    validate_motion_limits(motion, config.max_translation_cm, config.max_rotation_deg)
    target = build_target_pose(scene.hand_matrix(), paper_to_hand_motion(motion), tcp_z_offset_m=TCP_OFFSET)
    tcp = target[:3, 3] + TCP_OFFSET * target[:3, 2]
    if not config.contains_tcp(tcp):
        raise ValueError('Target outside the TCP workspace specified by --config')
    width = motion.gripper.state * scene.config.gripper_max_width_m
    if motion.gripper.state >= .5:
        scene.controller.gripper(width)
    if motion.translation.distance_cm or motion.rotation.angle_deg:
        scene.controller.move_pose(target[:3, 3], rot_matrix_to_quat(target[:3, :3]), duration=config.action_duration_s)
    if motion.gripper.state < .5:
        scene.controller.gripper(width)
    scene.step_seconds(config.action_settle_s)
    return target


def build_parser():
    parser = arguments(__doc__)
    parser.set_defaults(output=PROJECT_ROOT / 'output/vlmasam' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--task', choices=['pick_place', 'close_laptop'], default='pick_place')
    parser.add_argument('--instruction', help='Override the language instruction; evaluation remains task-specific')
    parser.add_argument('--system-prompt', type=Path, help='Optional fixed policy sent once before conversation history')
    parser.add_argument('--label-images', action='store_true', help='Attach source step and camera names to each image without changing pixels')
    parser.add_argument('--prompt-template', type=Path, default=PROMPT_PATH, help='Prompt template to snapshot for this run')
    parser.add_argument('--model', default='openai/gpt-6-astra')
    parser.add_argument('--reasoning-effort', choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'], default=DEFAULT_REASONING_EFFORT, help='Default xhigh; must be supported by the model')
    parser.add_argument('--max-tokens', type=int, default=None, help='Response budget including reasoning; default 4096 for Qwen3.8-27B, 350 otherwise')
    parser.add_argument('--base-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--keys-file', type=Path, help='Optional private JSON list containing exactly one API key, never copied into output')
    parser.add_argument('--image-encoding', choices=['png', 'jpeg'], default=DEFAULT_IMAGE_ENCODING, help='JPEG quality 90 at original resolution avoids the provider image-size cap for full history')
    parser.add_argument('--max-task-steps', type=int, default=35)
    parser.add_argument('--chat-connection-mode', choices=CHAT_CONNECTION_MODES, default='separate',
                        help='separate: current step only; continuous: all history; first-recent: first plus latest N turns, with images')
    parser.add_argument('--chat-recent-turns', type=int, default=1, help='Recent completed turns for first-recent (at least 1)')
    parser.add_argument('--observe-only', action='store_true')
    return parser


def configure_args(parser, args):
    args.scene_config = config_from_args(parser, args)
    try:
        template = load_prompt(args.prompt_template, args.scene_config, system_path=args.system_prompt)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.label_images = args.label_images or template.system is not None
    if args.max_tokens is None:
        args.max_tokens = default_max_tokens(args.model)
    if args.chat_recent_turns < 1:
        parser.error('--chat-recent-turns must be at least 1')
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    if args.instruction is None:
        args.instruction = ('Close the laptop lid completely, then withdraw the robot hand.' if args.task == 'close_laptop'
                            else 'Pick up the red cube, raise it at least 8 cm above its starting support, then place it at the CENTER of the blue square pad. Release it and retreat at least 8 cm. Verify the cube is fully inside the pad with margin on every side, not on its edge.')
    return template


@contextmanager
def simulation_session(args, template, *, use_client=True, metadata=None):
    """Own simulation lifetime and shared setup for task entry points."""
    client = None if not use_client or args.observe_only else VisionClient(args.model, args.base_url, args.keys_file, max_tokens=args.max_tokens,
        chat_connection_mode=args.chat_connection_mode, chat_recent_turns=args.chat_recent_turns, image_encoding=args.image_encoding, reasoning_effort=args.reasoning_effort, system_prompt=template.system, label_images=args.label_images)
    app = start_app(args)
    scene = None
    try:
        if args.task == 'close_laptop':
            from construct_scene.laptop_scene import LaptopScene
            scene = LaptopScene(app, args.assets, config=args.scene_config)
        else:
            scene = ManipulationScene(app, args.assets, config=args.scene_config)
        scene.prepare(args.output)
        (args.output / 'physics_config.json').write_text(json.dumps(args.scene_config.to_dict(), indent=2))
        scene.prompt_template = template
        (args.output / 'prompt_template.txt').write_text(template.observation)
        (args.output / 'prompt_source.txt').write_text(template.source)
        if template.system is not None:
            (args.output / 'system_prompt.txt').write_text(template.system)
        (args.output / 'run_config.json').write_text(json.dumps({
            'system_prompt_file': str(args.system_prompt.resolve()) if args.system_prompt else (str(args.prompt_template.resolve()) if template.system else None),
            'prompt_source_snapshot': str((args.output / 'prompt_source.txt').resolve()),
            'system_prompt_snapshot': str((args.output / 'system_prompt.txt').resolve()) if template.system else None, 'image_source_labels': args.label_images,
            'prompt_file': str(args.prompt_template.resolve()), 'prompt_snapshot': str((args.output / 'prompt_template.txt').resolve()), 'motion_format': 'pipe', 'motion_frame': TCP_FRAME,
            'config': args.config, 'physics_config': args.scene_config.to_dict(),
            'task': args.task, 'model': args.model, 'base_url': args.base_url, 'instruction': args.instruction,
            'gpu': args.gpu, 'runtime': scene.runtime, 'assets': str(scene.assets), 'mode': 'observation' if args.observe_only else 'closed_loop',
            'chat_connection_mode': args.chat_connection_mode, 'chat_recent_turns': args.chat_recent_turns,
            'image_encoding': args.image_encoding, 'jpeg_quality': 90 if args.image_encoding == 'jpeg' else None, 'image_resize': False,
            'reasoning_effort': args.reasoning_effort, 'max_tokens': args.max_tokens, 'physics_dt': 1/120, 'tcp_offset_m': TCP_OFFSET, 'max_task_steps': args.max_task_steps,
            **(metadata or {}),
        }, indent=2))
        yield scene, client
    except BaseException:
        import traceback
        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        try:
            if scene is not None and getattr(scene, 'recording', False):
                scene.finish()
        finally:
            app.close()


def run_loop(args, scene, client, *, start_index=1,
             previous='Ready; gripper open. No action has been executed yet.'):
    for index in range(start_index, args.max_task_steps + 1):
        if (args.output / 'STOP').exists():
            print('Trial stopped by output/STOP; saving current physical state.', flush=True)
            capture(scene, args.output / 'final')
            break
        folder = args.output / f'step_{index:03d}'
        info, images = capture(scene, folder)
        prompt = prompt_for(scene, info, args.instruction, previous)
        (folder / 'prompt.txt').write_text(prompt)
        if args.observe_only:
            break
        print(f'VLM STEP {index}: querying {args.model}', flush=True)
        response = client.chat(prompt, images, context_path=folder / 'context.json', step_index=index)
        (folder / 'response.txt').write_text(response)
        (args.output / 'api_usage.json').write_text(json.dumps(client.usage, indent=2))
        print(response, flush=True)
        try:
            motion = parse_vlm_motion(response)
            validate_motion_limits(motion, scene.config.max_translation_cm, scene.config.max_rotation_deg)
            (folder / 'action.json').write_text(json.dumps(motion_to_dict(motion), indent=2))
            if motion.task_state == 1:
                scene.step_seconds(scene.config.completion_settle_s)
                previous = 'Completion was rejected by independent evaluation. Inspect the images. ' + getattr(scene, 'completion_feedback', 'The cube must be lifted, carried onto the blue pad, released, and the hand moved clear.')
            else:
                target = execute(scene, motion)
                (folder / 'target_hand_world.json').write_text(json.dumps(target.tolist()))
                measured_width = float(sum(scene.robot.get_joint_positions()[scene.finger_indices]))
                requested_width = motion.gripper.state * scene.config.gripper_max_width_m
                previous = ('Motion step ended. Requested gripper opening: %.1f mm; measured opening: %.1f mm. '
                            'Re-observe the current images.') % (requested_width * 1000, measured_width * 1000)
                if motion.gripper.state >= .5 and measured_width + .005 < requested_width:
                    previous += ' The requested opening has NOT been reached; do not assume release succeeded.'
        except (ValueError, RuntimeError) as error:
            previous = str(error)
            print('EXECUTION FEEDBACK:', previous, flush=True)
            (folder / 'execution_error.txt').write_text(previous)
        evaluation = scene.evaluate()
        (folder / 'evaluation.json').write_text(json.dumps(evaluation, indent=2))
        if evaluation['success']:
            print('PHYSICAL TASK SUCCESS:', args.task, flush=True)
            capture(scene, args.output / 'final')
            break


def finish_run(args, scene):
    # Isaac Sim fast shutdown can terminate the process in app.close().
    # Save and validate the result while simulation_session still handles errors.
    if getattr(scene, 'recording', False):
        scene.finish()
    result = scene.result
    if not args.observe_only and not result['success']:
        completed_turns = len(list(args.output.glob('step_*/response.txt')))
        raise RuntimeError(f'Task did not succeed; {completed_turns} replies recorded (limit {args.max_task_steps}); inspect {args.output}')
    print('RESULT', json.dumps(result), flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    templates = configure_args(parser, args)
    with simulation_session(args, templates) as (scene, client):
        run_loop(args, scene, client)
        finish_run(args, scene)


if __name__ == '__main__':
    main()
