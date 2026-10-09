"""Project true TCP axes and estimate axial clearance from rendered depth only."""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_tcp_axes(canvas, projected):
    """Draw full-frame signed X/Y guides and a finite +Z depth-surface arrow."""
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    origin = np.asarray(projected[0], dtype=float)
    if not np.all(np.isfinite(origin)):
        return
    width, height = canvas.size
    directions = []
    labels = []
    for index, endpoint in enumerate(projected[1:3]):
        vector = np.asarray(endpoint, dtype=float) - origin
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm < 1e-4:
            # An axis seen exactly end-on has no screen-space direction.
            continue
        direction = vector / norm
        low, high = -np.inf, np.inf
        for axis, limit in enumerate((width, height)):
            if abs(direction[axis]) < 1e-9:
                if not 1 <= origin[axis] <= limit - 2:
                    low, high = 1, 0
                    break
            else:
                ends = sorted(((1 - origin[axis]) / direction[axis],
                               (limit - 2 - origin[axis]) / direction[axis]))
                low, high = max(low, ends[0]), min(high, ends[1])
        if low >= high:
            continue
        color = ('#ff454f', '#24df72', '#359dff')[index]
        overlap = any(abs(np.dot(direction, prior)) > .985 for prior in directions)
        directions.append(direction)
        start, end = origin + low * direction, origin + high * direction
        before_y = canvas.copy() if index == 1 else None
        if overlap:
            # Preserve coincident projections with narrower dashes rather than
            # rotating or offsetting either physical axis.
            for distance in np.arange(0, high - low, 18):
                a = start + direction * distance
                b = start + direction * min(distance + 10, high - low)
                draw.line([tuple(a), tuple(b)], fill=color, width=3)
        else:
            draw.line([tuple(start), tuple(end)], fill=color, width=4)
        # Only fade the Y stroke where it coincides with the finite Z segment.
        # Preserve all line widths, arrows, labels, and the true projected points.
        if before_y is not None and len(projected) > 3 and np.all(np.isfinite(projected[3])):
            z_vector = np.asarray(projected[3], dtype=float) - origin
            z_length = np.linalg.norm(z_vector)
            if z_length > 1 and abs(np.dot(direction, z_vector / z_length)) > .985:
                z_direction = z_vector / z_length
                yy, xx = np.indices((height, width))
                dx, dy = xx - origin[0], yy - origin[1]
                along = dx*z_direction[0] + dy*z_direction[1]
                across = np.abs(dx*z_direction[1] - dy*z_direction[0])
                mask = ((along >= 0) & (along <= z_length) & (across <= 4))
                faded = Image.blend(before_y, canvas, .20)
                canvas.paste(faded, (0, 0), Image.fromarray(mask.astype('uint8') * 255))
        for sign, distance in ((-1, low), (1, high)):
            if sign * distance <= 0:
                continue
            tip = origin + distance * direction
            outward = sign * direction
            normal = np.array([-outward[1], outward[0]])
            arrow_length, arrow_half_width = (10, 4) if overlap else (16, 7)
            draw.polygon([tuple(tip), tuple(tip - arrow_length*outward + arrow_half_width*normal),
                          tuple(tip - arrow_length*outward - arrow_half_width*normal)], fill=color)
            label = ('+' if sign > 0 else '-') + 'XYZ'[index]
            box = draw.textbbox((0, 0), label, font=font)
            bw, bh = box[2]-box[0]+12, box[3]-box[1]+10
            anchor = tip - 26*outward
            # Put labels of coincident axes on opposite sides of the common line.
            side = 1 if overlap else -1
            if abs(direction[1]) >= abs(direction[0]):
                left = anchor[0]+12 if side > 0 else anchor[0]-bw-12
                top = anchor[1]-bh/2
            else:
                left = anchor[0]-bw/2
                top = anchor[1]+12 if side > 0 else anchor[1]-bh-12
            left = float(np.clip(left, 5, max(5, width-bw-5)))
            top = float(np.clip(top, 5, max(5, height-bh-5)))
            labels.append((left, top, bw, bh, box, label, color))
    # Paper convention: Z terminates at the selected visible depth sample.
    # Never extend Z to an image edge or invent a fixed-length depth estimate.
    if len(projected) > 3 and np.all(np.isfinite(projected[3])):
        tip = np.asarray(projected[3], dtype=float)
        vector = tip - origin
        length = np.linalg.norm(vector)
        color = '#359dff'
        if length > 1:
            direction = vector / length
            normal = np.array([-direction[1], direction[0]])
            draw.line([tuple(origin), tuple(tip)], fill=color, width=4)
            size = min(14., length * .4)
            draw.polygon([tuple(tip), tuple(tip-size*direction+size*.4*normal),
                          tuple(tip-size*direction-size*.4*normal)], fill=color)
        if 0 <= tip[0] < width and 0 <= tip[1] < height:
            draw.ellipse((tip[0]-3, tip[1]-3, tip[0]+3, tip[1]+3), fill=color)
            label = '+Z'
            box = draw.textbbox((0, 0), label, font=font)
            bw, bh = box[2]-box[0]+12, box[3]-box[1]+10
            left = float(np.clip(tip[0]+12, 5, max(5, width-bw-5)))
            top = float(np.clip(tip[1]-bh-6, 5, max(5, height-bh-5)))
            labels.append((left, top, bw, bh, box, label, color))
    for left, top, bw, bh, box, label, color in labels:
        draw.rounded_rectangle((left, top, left+bw, top+bh), radius=5, fill='#202328')
        draw.text((left+6-box[0], top+5-box[1]), label, font=font, fill=color)
    if 0 <= origin[0] < width and 0 <= origin[1] < height:
        x, y = origin
        draw.ellipse((x-5, y-5, x+5, y+5), fill='white', outline='black', width=1)
        draw.ellipse((x-2, y-2, x+2, y+2), fill='black')


def capture(scene, folder):
    folder.mkdir(parents=True, exist_ok=True)
    # Update rendered observations while preserving the physics state during API calls.
    for _ in range(8):
        scene.world.render()
    rgb = scene.wrist.get_rgba()[:, :, :3].astype('uint8')
    depth = scene.wrist.get_depth()
    if depth is None or not np.any(np.isfinite(depth) & (depth > 0)):
        raise RuntimeError('Wrist RGB-D camera did not produce valid depth')
    Image.fromarray(rgb).save(folder / 'wrist.png')
    np.save(folder / 'depth_m.npy', depth)
    Image.fromarray(scene.overview.get_rgba()[:, :, :3].astype('uint8')).save(folder / 'overview.png')
    Image.fromarray(scene.camera.get_rgba()[:, :, :3].astype('uint8')).save(folder / 'external.png')
    T = scene.tcp_matrix()
    yy, xx = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > .015) & (depth < 2.)
    pixels = np.column_stack([xx[valid], yy[valid]])
    points = scene.wrist.get_world_points_from_image_coords(pixels, depth[valid])
    delta = points - T[:3, 3]
    axial = delta @ T[:3, 2]
    radial = np.linalg.norm(delta - axial[:, None] * T[:3, 2], axis=1)
    candidates = np.flatnonzero((radial < .006) & (axial > .002) & (axial < .8))
    clearance, surface_point = None, None
    if len(candidates) >= 3:
        # Select an actual visible sample at the lower 5th percentile, avoiding a
        # single near outlier. Its axial distance and drawn endpoint stay paired.
        ordered = candidates[np.argsort(axial[candidates])]
        selected = ordered[int(.05 * (len(ordered)-1))]
        surface_point = points[selected]
        clearance = float(axial[selected])
    markers = [T[:3, 3], T[:3, 3] + .05*T[:3, 0], T[:3, 3] + .05*T[:3, 1]]
    if surface_point is not None:
        markers.append(surface_point)
    uv = scene.wrist.get_image_coords_from_world_points(np.array(markers))
    canvas = Image.fromarray(rgb)
    draw_tcp_axes(canvas, uv)
    canvas.save(folder / 'wrist_tcp.png')
    # The fixed view also shows the SAME physical TCP axes. This disambiguates
    # horizontal transport; overhead views are saved only for inspection.
    # The model receives the external view, annotated wrist, and raw wrist.
    top = Image.open(folder / 'overview.png').convert('RGB')
    top_points = np.array([T[:3, 3], T[:3, 3] + .10 * T[:3, 0], T[:3, 3] + .10 * T[:3, 1]])
    top_uv = scene.overview.get_image_coords_from_world_points(top_points)
    draw_tcp_axes(top, top_uv)
    top.save(folder / 'overview_tcp.png')
    images = [folder / 'external.png', folder / 'wrist_tcp.png', folder / 'wrist.png']
    width = float(sum(scene.robot.get_joint_positions()[scene.finger_indices]))
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    camera_position, camera_quat = scene.wrist.get_world_pose(camera_axes='ros')
    camera_rotation = quat_to_rot_matrix(camera_quat)
    tcp_camera = camera_rotation.T @ (T[:3, 3] - camera_position)
    tcp_axes_camera = camera_rotation.T @ T[:3, :3]
    info = {'observation_id': folder.name, 'vlm_image_order': [path.name for path in images], 'tcp_world': T.tolist(), 'wrist_K': scene.wrist.get_intrinsics_matrix().tolist(),
            'wrist_resolution': [int(rgb.shape[1]), int(rgb.shape[0])],
            'tcp_in_wrist_camera_m': tcp_camera.tolist(), 'tcp_axes_in_wrist_camera': tcp_axes_camera.tolist(),
            'clearance_m': clearance,
            'z_surface_point_world': None if surface_point is None else surface_point.tolist(),
            'gripper_width_m': width,
            'tcp_height_above_workmat_m': float(T[2, 3] - scene.config.workmat_top_m),
            'tcp_projected_pixels': uv.tolist(),
            'motion_frame': 'paper_tcp_v1',
            'axis_overlay': 'X/Y signed direction guides extend to edges; +Z ends at a visible RGB-D surface sample, omitted if unavailable.',
            'source': 'RGB-D and robot proprioception only'}
    (folder / 'observation.json').write_text(json.dumps(info, indent=2))
    return info, images
