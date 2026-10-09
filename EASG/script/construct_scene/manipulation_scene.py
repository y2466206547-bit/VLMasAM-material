"""RoboDojo block, physical grasping, calibrated wrist RGB-D and task evaluation."""
import json
import math
from pathlib import Path
import numpy as np
from construct_scene.init_scene import TabletopScene
from control_robot import RobotController

TCP_OFFSET = 0.095  # panda_hand -> centre of the finger contact pads, metres


class ManipulationScene(TabletopScene):
    include_tabletop_props = False

    def add_task_objects(self):
        from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        from isaacsim.sensors.camera import Camera

        for name in ('Floor', 'BackWall', 'SideWall', 'Skirting'):
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath('/World/' + name)).CreateCollisionEnabledAttr(False)
        # A fixed table is a kinematic rigid body with colliders, not a falling dynamic body.
        body = UsdPhysics.RigidBodyAPI.Apply(self.stage.GetPrimAtPath('/World/Table'))
        body.CreateKinematicEnabledAttr(True)

        self.add_manipulandum()
        rubber = UsdPhysics.MaterialAPI.Apply(self.stage.DefinePrim('/World/Looks/FingerPhysics', 'Material'))
        rubber.CreateStaticFrictionAttr(1.2)
        rubber.CreateDynamicFrictionAttr(1.0)
        rubber.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(rubber.GetPrim()).CreateFrictionCombineModeAttr('max')
        self.finger_contacts = []
        for side in ('left', 'right'):
            path = f'/World/Robot/fr3/panda_{side}finger'
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(path)):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(UsdShade.Material(rubber.GetPrim()),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(p)
                    collision.CreateContactOffsetAttr(0.002)
                    collision.CreateRestOffsetAttr(0.0)
            view = RigidPrim(path, name=side + '_contact', track_contact_forces=True,
                                 contact_filter_prim_paths_expr=self.contact_targets)
            self.finger_contacts.append(self.world.scene.add(view))
        self.overview = self.world.scene.add(Camera('/World/TaskOverview', name='overview',
            position=np.array([0.08, 0., self.table_height + 1.0]), orientation=np.array([0., 1., 0., 0.]),
            resolution=(640, 480), frequency=30))
        self.overview.set_world_pose(np.array([0.08, 0., self.table_height + 1.0]), np.array([0., 1., 0., 0.]), camera_axes='ros')
        self.overview.prim.GetAttribute('focalLength').Set(20.)
        self.overview.set_clipping_range(.02, 10.)

        self.wrist = self.world.scene.add(Camera('/World/Robot/fr3/panda_hand/WristCamera',
                                                name='wrist', resolution=(640, 480), frequency=30))
        # ROS optical frame in panda_hand coordinates: forward along the fingers,
        # image right along jaw separation, image down toward hand -X.
        # Side offset frames the jaws below centre; the wide lens keeps both visible.
        eye = np.array([0.08, 0., 0.030])
        x = np.array([0., 1., 0.])
        y = np.array([-1., 0., 0.])
        z = np.array([0., 0., 1.])
        self.wrist.set_local_pose(eye, rot_matrix_to_quat(np.column_stack([x, y, z])), camera_axes='ros')
        self.wrist.prim.GetAttribute('focalLength').Set(5.0)
        self.wrist.prim.GetAttribute('horizontalAperture').Set(20.955)
        self.wrist.prim.GetAttribute('verticalAperture').Set(15.71625)
        self.wrist.set_clipping_range(0.015, 10)

    def add_manipulandum(self):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        self.contact_targets = ['/World/Block']
        self.goal_xy = np.array([0.15, 0.12])
        self.block_size = 0.035
        self.pad_top = self.table_height + 0.008
        self.box('GoalPad', (*self.goal_xy, self.table_height + 0.006), (0.15, 0.15, 0.004),
                 self.material('GoalBlue', (0.015, 0.15, 0.75), roughness=0.85))
        prim = add_reference_to_stage(str(self.asset('objects/rigid/cube/object.usdz')), '/World/Block')
        self.block = self.world.scene.add(SingleRigidPrim('/World/Block', name='block',
                                                   position=np.array([-0.04, -0.12, self.table_height + 0.027]), mass=0.1))
        red = self.material('BlockRed', (0.65, 0.025, 0.018), roughness=0.45)
        # Keep the packaged collision mesh; disable duplicate visual collisions.
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(red, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr('/collision/' in str(p.GetPath()))
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
        physics = UsdPhysics.MaterialAPI.Apply(self.stage.DefinePrim('/World/Looks/BlockPhysics', 'Material'))
        physics.CreateStaticFrictionAttr(0.3)  # RoboDojo cube metadata
        physics.CreateDynamicFrictionAttr(0.3)
        physics.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(physics.GetPrim()),
                  bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(64)
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(p)
                collision.CreateContactOffsetAttr(0.002)
                collision.CreateRestOffsetAttr(0.0)


    @property
    def initial_hand_position(self):
        return [-0.06, -0.03, self.table_height + 0.38]

    def prepare(self, output):
        self.wrist.add_distance_to_image_plane_to_frame()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.controller = getattr(self, 'controller_class', GraspController)(self)
        self.controller.move_pose(self.initial_hand_position, [0, 1, 0, 0], duration=self.config.initial_pose_duration_s)
        self.initialize_task()
        self.telemetry = []
        self.recording = True
        self.video = None
        import imageio.v2 as imageio
        self.video = imageio.get_writer(str(self.output / 'execution.mp4'), fps=15, codec='libx264', quality=7)
        self.record_sample()

    def initialize_task(self):
        self.initial_block = self.block.get_world_pose()[0].copy()
        self.max_lift = 0.0
        self.bilateral_contact_seen = False

    def hand_matrix(self):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        position, quat = self.hand.get_world_pose()
        T = np.eye(4); T[:3, :3] = quat_to_rot_matrix(quat); T[:3, 3] = position
        return T

    def tcp_matrix(self):
        from motion import paper_tcp_pose
        return paper_tcp_pose(self.hand_matrix(), TCP_OFFSET)

    def contacts(self):
        return [float(np.linalg.norm(v.get_contact_force_matrix(dt=1 / 120))) for v in self.finger_contacts]

    def record_sample(self):
        p, q = self.block.get_world_pose()
        forces = self.contacts()
        lift = float(p[2] - self.initial_block[2])
        self.max_lift = max(self.max_lift, lift)
        if min(forces) > 0.1 and lift > 0.035:
            self.bilateral_contact_seen = True
        self.telemetry.append({'time': self.tick / 120, 'block': p.tolist(), 'block_quaternion': q.tolist(),
             'tcp': self.tcp_matrix()[:3, 3].tolist(), 'finger_contact_N': forces,
             'gripper_width': float(sum(self.robot.get_joint_positions()[self.finger_indices]))})
        if self.video is not None:
            rgba = self.camera.get_rgba()
            if rgba is not None and rgba.size:
                self.video.append_data(rgba[:, :, :3].astype('uint8'))

    def step(self, count=1):
        for _ in range(count):
            super().step()
            if getattr(self, 'recording', False) and self.tick % 8 == 0:
                self.record_sample()

    def evaluate(self):
        p = self.block.get_world_pose()[0]
        velocity = self.block.get_linear_velocity()
        width = float(sum(self.robot.get_joint_positions()[self.finger_indices]))
        target_error = float(np.linalg.norm(p[:2] - self.goal_xy))
        placed = target_error < 0.05 and abs(p[2] - (self.pad_top + self.block_size / 2)) < 0.012
        separated = np.linalg.norm(self.tcp_matrix()[:3, 3] - p) > 0.07
        success = self.max_lift > 0.05 and self.bilateral_contact_seen and placed and width > self.config.gripper_max_width_m * (13 / 16) and separated and np.linalg.norm(velocity) < 0.02
        return {'success': bool(success), 'max_lift_m': self.max_lift,
                'bilateral_contact_while_lifted': self.bilateral_contact_seen,
                'initial_block_position': self.initial_block.tolist(), 'final_block_position': p.tolist(),
                'goal_xy': self.goal_xy.tolist(), 'goal_error_m': target_error,
                'released': width > self.config.gripper_max_width_m * (13 / 16), 'hand_clear': bool(separated),
                'final_speed_m_s': float(np.linalg.norm(velocity)),
                'physics_only': True, 'object_poses_sent_to_vlm': False}

    def finish(self):
        if not getattr(self, 'recording', False):
            return self.result
        self.recording = False
        if self.video is not None:
            self.video.close(); self.video = None
        (self.output / 'telemetry.json').write_text(json.dumps(self.telemetry, indent=2))
        self.result = self.evaluate()
        (self.output / 'result.json').write_text(json.dumps(self.result, indent=2))
        return self.result


class GraspController(RobotController):
    """Position-drive grasp: allow fingers to stop at contact while retaining force."""
    def gripper(self, width, duration=None):
        from isaacsim.core.utils.types import ArticulationAction
        maximum = self.scene.config.gripper_max_width_m
        duration = self.scene.config.gripper_duration_s if duration is None else duration
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('duration must be positive and finite')
        if not np.isfinite(width) or not 0 <= width <= maximum:
            raise ValueError(f'width must be in [0, {maximum}]')
        indices = self.scene.finger_indices
        self.robot.get_articulation_controller().set_max_efforts(np.array([35., 35.]), joint_indices=indices)
        if abs(width - getattr(self, 'desired_width', maximum)) < 1e-6:
            # Re-issuing close must NOT ramp from measured contact width: that
            # temporarily removes the spring force and drops the held object.
            self.scene.step_seconds(duration * 0.25)
            return {'requested_width': width, 'measured_width': float(sum(self.robot.get_joint_positions()[indices]))}
        self.desired_width = width
        start = self.robot.get_joint_positions()[indices]
        target = np.array([width / 2, width / 2])
        steps = max(1, math.ceil(duration / self.scene.world.get_physics_dt()))
        for i in range(steps):
            q = start + (target - start) * min(1., (i + 1) / (steps * 0.75))
            self.robot.apply_action(ArticulationAction(joint_positions=q, joint_indices=indices))
            self.scene.step()
        return {'requested_width': width, 'measured_width': float(sum(self.robot.get_joint_positions()[indices]))}
