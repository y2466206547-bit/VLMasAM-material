"""Build a standalone Isaac Sim tabletop. All lengths are metres; Z is up."""
import argparse
import json
import os
from pathlib import Path
import math
import sys

# Permit the documented direct script invocation as well as package imports.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_config, config_from_args

# Shared scripts directory: task configuration and prompts remain here.
HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parent
DEFAULT_ASSETS = PROJECT_ROOT / 'Asset'
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def arguments(description, *, demo=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='default.json', help='JSON filename in EASG/config')
    parser.add_argument('--assets', type=Path, default=Path(os.environ.get('EASG_ASSETS', os.environ.get('ROBOSCENE_ASSETS', DEFAULT_ASSETS))))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(4), default=0, help='GPU index for rendering and physics')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'output')
    parser.add_argument('--path-tracing', action='store_true', help='Slower, higher quality rendering')
    if demo:
        parser.add_argument('--steps', type=int, default=None, help='Extra simulation steps; GUI defaults to unlimited, headless to 120')
        parser.add_argument('--screenshot', action='store_true')
    return parser


def start_app(args):
    # Kit must start before importing Isaac APIs.
    from isaacsim import SimulationApp
    return SimulationApp({'headless': args.headless, 'width': 1280, 'height': 960,
                          'renderer': 'PathTracing' if args.path_tracing else 'RayTracedLighting',
                          'samples_per_pixel_per_frame': 32, 'multi_gpu': False,
                          'active_gpu': args.gpu, 'physics_gpu': args.gpu})


class TabletopScene:
    """Own the World, robot and camera; no imports from InternDataEngine."""

    include_tabletop_props = True

    def __init__(self, app, assets, config=None):
        import numpy as np
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera

        self.config = config if config is not None else load_config()
        self.app, self.assets = app, Path(assets).expanduser().resolve()
        import sys
        from importlib.metadata import version
        self.runtime = {'isaacsim_version': version('isaacsim'), 'python_executable': sys.executable}
        print('Simulation runtime:', self.runtime, flush=True)
        self.tick = 0
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
        self.stage = self.world.stage
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath('/World'))
        self.table_height = self.config.table_height_m
        self.objects = {}

        floor = self.material('Floor', (0.27, 0.29, 0.30), roughness=0.8)
        wall = self.material('Wall', (0.70, 0.72, 0.70), roughness=0.9)
        rubber = self.material('Rubber', (0.045, 0.065, 0.07), roughness=0.85)
        metal = self.material('Metal', (0.32, 0.34, 0.36), metallic=0.8, roughness=0.3)
        self.box('Floor', (0, 0, -0.06), (8, 8, 0.12), floor)
        self.box('BackWall', (-1.8, 0, 1.5), (0.12, 8, 3), wall)
        self.box('SideWall', (0, 2.2, 1.5), (4, 0.12, 3), wall)
        self.box('Skirting', (-1.72, 0, 0.06), (0.04, 8, 0.12), metal)
        self.place_asset('Table', 'environment/table0/instance.usd', (0, 0), 0,
                         dimensions=(1.6, 0.9, self.table_height), dynamic=False)
        self.box('WorkMat', (0.20, 0, self.table_height + 0.002), (0.65, 0.59, 0.004), rubber)
        self.box('RobotMount', (-0.50, 0, self.table_height + 0.015), (0.23, 0.24, 0.03), metal)

        if self.include_tabletop_props:
            source = 'objects/rigid'
            self.place_asset('Mug', f'{source}/omniobject3d-mug/google_scan-mug_0160/Aligned_obj.usd',
                             (0.13, -0.15), self.table_height + 0.005, height=0.115, rotation=(90, 0, 0))
            for name, category, xy, height, rotation in [
                ('Banana', 'omniobject3d-banana', (0.36, 0.13), 0.065, (0, 0, 0)),
                ('Book', 'google_scan-book', (0.65, -0.17), 0.025, (90, 0, 12)),
            ]:
                candidates = sorted((self.assets / source / category).glob('*/Aligned_obj.usd'))
                if not candidates:
                    raise FileNotFoundError(f'No USD assets in {self.assets / source / category}')
                self.place_asset(name, candidates[0], xy, self.table_height + 0.005, height=height, rotation=rotation,
                                 dimensions=(0.22, 0.16, 0.025) if name == 'Book' else None)

        robot_path = '/World/Robot'
        add_reference_to_stage(str(self.asset('robot/franka/robot.usd')), robot_path)
        SingleXFormPrim(robot_path).set_world_pose(position=np.array([-0.50, 0, self.table_height + 0.03]))
        self.robot = self.world.scene.add(Robot(prim_path=robot_path, name='franka'))
        self.base = SingleXFormPrim(robot_path + '/fr3/panda_link0')
        self.hand = SingleXFormPrim(robot_path + '/fr3/panda_hand')
        # Explicit contact material on the table and props, authored only in our stage.
        contact = UsdPhysics.MaterialAPI.Apply(self.stage.DefinePrim('/World/Looks/Contact', 'Material'))
        contact.CreateStaticFrictionAttr(0.7)
        contact.CreateDynamicFrictionAttr(0.5)
        contact.CreateRestitutionAttr(0.02)
        from pxr import UsdShade
        for name in self.objects:
            UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath('/World/' + name)).Bind(
                UsdShade.Material(contact.GetPrim()), materialPurpose='physics')

        dome = UsdLux.DomeLight.Define(self.stage, '/World/Lights/Environment')
        dome.CreateTextureFileAttr(str(self.asset('environment/envmap_lib/studio_small_03_1k.hdr')))
        dome.CreateIntensityAttr(250)
        dome.CreateTextureFormatAttr('latlong')
        key = UsdLux.RectLight.Define(self.stage, '/World/Lights/Softbox')
        key.CreateWidthAttr(2.2)
        key.CreateHeightAttr(1.5)
        key.CreateIntensityAttr(500)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.83))
        UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.3, -0.3, 2.7))

        # Move the external camera 35% closer along the same viewing ray.
        eye, target = (1.365, -1.56, self.table_height + 0.8475), (0, 0, self.table_height + 0.10)
        camera_path = '/World/Camera'
        self.camera = self.world.scene.add(Camera(camera_path, resolution=(1280, 960), frequency=30))
        transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        q = transform.ExtractRotationQuat()
        self.camera.set_world_pose(np.array(eye), np.array([q.GetReal(), *q.GetImaginary()]), camera_axes='usd')
        self.camera.prim.GetAttribute("focalLength").Set(28.0)  # USD tenths of a stage unit
        self.camera.set_clipping_range(0.02, 100)
        set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path='/OmniverseKit_Persp')
        self.add_task_objects()
        self.world.reset()
        self.robot.set_solver_position_iteration_count(64)
        self.robot.set_solver_velocity_iteration_count(4)
        self.arm_indices = np.array([self.robot.get_dof_index(f'panda_joint{i}') for i in range(1, 8)])
        self.finger_indices = np.array([self.robot.get_dof_index(f'panda_finger_joint{i}') for i in (1, 2)])
        q = self.robot.get_joint_positions()
        q[self.arm_indices], q[self.finger_indices] = HOME, self.config.gripper_max_width_m / 2
        self.robot.set_joints_default_state(positions=q)
        self.robot.set_joint_positions(q)  # Teleport ONLY at initialization.
        self.robot.get_articulation_controller().set_gains(
            kps=np.array([6000.] * 7 + [800., 800.]), kds=np.array([180.] * 7 + [40., 40.]))
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(ArticulationAction(joint_positions=q))
        self.step(180)
        print('Scene ready. DOFs:', self.robot.dof_names, flush=True)

    def add_task_objects(self):
        """Subclass hook: add sensors and task bodies before physics initialization."""

    def asset(self, relative):
        path = self.assets / relative
        if not path.is_file():
            raise FileNotFoundError(f'Asset not found: {path}. Set --assets or EASG_ASSETS.')
        return path

    def material(self, name, color, roughness=0.5, metallic=0.0):
        from pxr import Gf, Sdf, UsdShade
        material = UsdShade.Material.Define(self.stage, '/World/Looks/' + name)
        shader = UsdShade.Shader.Define(self.stage, material.GetPath().AppendChild('Surface'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        return material

    def box(self, name, position, size, material):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
        cube = UsdGeom.Cube.Define(self.stage, '/World/' + name)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    def place_asset(self, name, relative, xy, bottom, height=None, dimensions=None, dynamic=True, rotation=(0, 0, 0)):
        """Fit bounds in metres, rest on a surface; preserve referenced meshes/materials."""
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from isaacsim.core.utils.stage import add_reference_to_stage
        root = UsdGeom.Xform.Define(self.stage, '/World/' + name)
        orient = UsdGeom.Xform.Define(self.stage, str(root.GetPath()) + '/Orientation')
        orient.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        model = add_reference_to_stage(str(self.asset(relative)), str(orient.GetPath()) + '/Model')
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(model).ComputeAlignedRange()
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        size = high - low
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError(f'Invalid bounds for {relative}: {size}')
        scale = np.array(dimensions) / size if dimensions else np.full(3, height / size[2])
        center = (low + high) / 2 * scale
        root.AddTranslateOp().Set(Gf.Vec3d(float(xy[0] - center[0]), float(xy[1] - center[1]), float(bottom - low[2] * scale[2] + (0.002 if dynamic else 0))))
        root.AddScaleOp().Set(Gf.Vec3d(*scale))
        # Remove imported rigid-body behavior on descendants so there is one body per prop.
        for prim in Usd.PrimRange(model):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                prim.RemoveAPI(UsdPhysics.MassAPI)
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                normals = mesh.GetNormalsAttr().Get()
                if normals is not None and len(normals) == len(mesh.GetFaceVertexIndicesAttr().Get()):
                    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.15)
        self.objects[name] = {'xy': list(xy), 'height': float(size[2] * scale[2]), 'bottom': bottom}

    def step(self, count=1):
        for _ in range(count):
            if not self.app.is_running():
                raise RuntimeError('Simulation window closed')
            self.tick += 1
            self.world.step(render=self.tick % 4 == 0)

    def step_seconds(self, seconds):
        self.step(max(1, math.ceil(seconds / self.world.get_physics_dt())))

    def save(self, output, screenshot=False):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        self.stage.GetRootLayer().Export(str(output / 'tabletop.usda'))
        (output / 'scene.json').write_text(json.dumps({'runtime': self.runtime, 'assets': str(self.assets), 'objects': self.objects,
                                                       'physics_config': self.config.to_dict(), 'joint_names': self.robot.dof_names}, indent=2))
        if screenshot:
            from PIL import Image
            for _ in range(48):
                self.world.render()
            rgba = self.camera.get_rgba()
            if rgba is None or rgba.size == 0:
                raise RuntimeError('Camera produced no image')
            Image.fromarray(rgba.astype('uint8')).save(output / 'preview.png')
        print(f'Saved scene to {output}', flush=True)


def keep_running(scene, args):
    steps = args.steps if args.steps is not None else (120 if args.headless else None)
    if steps is not None:
        if steps < 0:
            raise ValueError('--steps must be nonnegative')
        scene.step(steps)
    else:
        while scene.app.is_running():
            scene.step()


def main():
    parser = arguments(__doc__, demo=True)
    args = parser.parse_args()
    config = config_from_args(parser, args)
    app = start_app(args)
    try:
        scene = TabletopScene(app, args.assets, config=config)
        scene.save(args.output, args.screenshot)
        keep_running(scene, args)
    except Exception:
        import traceback
        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
