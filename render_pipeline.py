import bpy
import bpy_extras
import json
import os
import random
import math

OUTPUT_DIR = "//results/render/"

SWIMMER_NAMES = [
    "Backstroke_man.mesh", "Backstroke_woman.mesh",
    "Breaststroke_man.mesh", "Breaststroke_woman.mesh",
    "Butterfly_man.mesh", "Butterfly_woman.mesh",
    "Crawl_man.mesh", "Crawl_woman.mesh",
    "Dive_man_left.mesh", "Dive_man_right.mesh",
    "Dive_woman_left.mesh", "Dive_woman_right.mesh",
    "Flip_man_left.mesh", "Flip_man_right.mesh",
    "Flip_woman_left.mesh", "Flip_woman_right.mesh"
]

KEYPOINT_MAPPING = {
    'left_hand': 'mixamorig:LeftHand',
    'right_hand': 'mixamorig:RightHand',
    'left_elbow': 'mixamorig:LeftForeArm',
    'right_elbow': 'mixamorig:RightForeArm',
    'left_shoulder': 'mixamorig:LeftArm',
    'right_shoulder': 'mixamorig:RightArm',
    'left_hip': 'mixamorig:LeftUpLeg',
    'right_hip': 'mixamorig:RightUpLeg',
    'left_knee': 'mixamorig:LeftLeg',
    'right_knee': 'mixamorig:RightLeg',
    'left_foot': 'mixamorig:LeftFoot',
    'right_foot': 'mixamorig:RightFoot'
}

FITZPATRICK_HEX = ['#FFDFC4', '#F0D5BE', '#D1A382', '#A67358', '#633922', '#361F13']

SWIMSUIT_PALETTES = [
    ('#0F0F0F', '#0F0F0F'),
    ('#1A2B4C', '#FFFFFF'),
    ('#8B0000', '#0F0F0F'),
    ('#004B87', '#FFD700'),
    ('#006400', '#FFD700'),
    ('#FFFFFF', '#1A2B4C'),
    ('#4B0082', '#E6E6FA'),
    ('#FF0000', '#FFFFFF'),
]

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    srgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    linear = tuple((c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)
    return linear + (1.0,)

def setup_gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.device = 'GPU'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = (device.type == 'OPTIX' or device.type == 'GPU')

def get_sequence_config(swimmer_name):
    name_lower = swimmer_name.lower()
    
    if "flip" in name_lower and "left" in name_lower:
        return 900, 1050, "Flip_cam"
    elif "dive" in name_lower and "left" in name_lower:
        return 926, 972, "Dive_cam"
    elif "flip" in name_lower and "right" in name_lower:
        return 1100, 1250, "Flip_cam"
    elif "dive" in name_lower and "right" in name_lower:
        return 1126, 1172, "Dive_cam"
    else:
        return 1, 850, "Swim_cam"

def hide_all_foreground_swimmers():
    for name in SWIMMER_NAMES:
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_render = True
            obj.hide_viewport = True
            
        rig_name = name.replace('.mesh', '')
        rig_obj = bpy.data.objects.get(rig_name)
        if rig_obj:
            rig_obj.hide_render = True
            rig_obj.hide_viewport = True

def toggle_background_swimmers(visible):
    for name in SWIMMER_NAMES:
        bg_mesh_name = f"{name}.001"
        bg_mesh = bpy.data.objects.get(bg_mesh_name)
        if bg_mesh:
            bg_mesh.hide_render = not visible
            bg_mesh.hide_viewport = not visible
            
        bg_rig_name = f"{name.replace('.mesh', '')}.001"
        bg_rig = bpy.data.objects.get(bg_rig_name)
        if bg_rig:
            bg_rig.hide_render = not visible
            bg_rig.hide_viewport = not visible

def set_foreground_materials():
    skin_hex = random.choice(FITZPATRICK_HEX)
    suit_hex, cap_hex = random.choice(SWIMSUIT_PALETTES)

    mat_skin = bpy.data.materials.get("skin")
    mat_suit = bpy.data.materials.get("swimsuit_body")
    mat_cap = bpy.data.materials.get("swimsuit_head")

    if mat_skin and mat_skin.use_nodes:
        mat_skin.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hex_to_rgba(skin_hex)
    if mat_suit and mat_suit.use_nodes:
        mat_suit.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hex_to_rgba(suit_hex)
    if mat_cap and mat_cap.use_nodes:
        mat_cap.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hex_to_rgba(cap_hex)

def randomize_background_materials():
    for i in range(1, 9):
        suffix = f"{i:03d}"
        skin_hex = random.choice(FITZPATRICK_HEX)
        suit_hex, cap_hex = random.choice(SWIMSUIT_PALETTES)

        mat_skin = bpy.data.materials.get(f"skin.{suffix}")
        mat_suit = bpy.data.materials.get(f"swimsuit_body.{suffix}")
        mat_cap = bpy.data.materials.get(f"swimsuit_head.{suffix}")

        if mat_skin and mat_skin.use_nodes:
            mat_skin.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hex_to_rgba(skin_hex)
        if mat_suit and mat_suit.use_nodes:
            mat_suit.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hex_to_rgba(suit_hex)
        if mat_cap and mat_cap.use_nodes:
            mat_cap.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hex_to_rgba(cap_hex)

def extract_keypoints(scene, cam, eval_armature, res_x, res_y):
    frame_data = {}

    for json_name, bone_name in KEYPOINT_MAPPING.items():
        pose_bone = eval_armature.pose.bones.get(bone_name)
        if not pose_bone:
            continue

        global_location = eval_armature.matrix_world @ pose_bone.head
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, global_location)

        frame_data[json_name] = {
            'x': round(co_2d.x * res_x),
            'y': round((1.0 - co_2d.y) * res_y),
            'in_front_of_cam': co_2d.z >= 0
        }

    bone_head = eval_armature.pose.bones.get('mixamorig:Head')
    if bone_head:
        pos_base = eval_armature.matrix_world @ bone_head.head
        pos_top = eval_armature.matrix_world @ bone_head.tail
        pos_center = (pos_base + pos_top) / 2.0
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos_center)
        frame_data['head'] = {
            'x': round(co_2d.x * res_x),
            'y': round((1.0 - co_2d.y) * res_y),
            'in_front_of_cam': co_2d.z >= 0
        }

    bone_l_shoulder = eval_armature.pose.bones.get('mixamorig:LeftArm')
    bone_r_shoulder = eval_armature.pose.bones.get('mixamorig:RightArm')
    if bone_l_shoulder and bone_r_shoulder:
        pos_l = bone_l_shoulder.head
        pos_l_global = eval_armature.matrix_world @ pos_l
        pos_r = bone_r_shoulder.head
        pos_r_global = eval_armature.matrix_world @ pos_r
        pos_sternum = (pos_l_global + pos_r_global) / 2.0
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos_sternum)
        frame_data['sternum'] = {
            'x': round(co_2d.x * res_x),
            'y': round((1.0 - co_2d.y) * res_y),
            'in_front_of_cam': co_2d.z >= 0
        }

    return frame_data

def main():
    setup_gpu()
    scene = bpy.context.scene

    render_scale = scene.render.resolution_percentage / 100.0
    res_x = scene.render.resolution_x * render_scale
    res_y = scene.render.resolution_y * render_scale

    out_dir_abs = bpy.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir_abs, exist_ok=True)

    for swimmer_name in SWIMMER_NAMES:
        start_frame, end_frame, camera_name = get_sequence_config(swimmer_name)
        
        cam = bpy.data.objects.get(camera_name)
        if not cam:
            continue
            
        scene.camera = cam

        original_rotation = cam.rotation_euler.copy()
        
        angle_noise_limit = math.radians(1.0) 
        
        cam.rotation_euler.x += random.uniform(-angle_noise_limit, angle_noise_limit)
        cam.rotation_euler.y += random.uniform(-angle_noise_limit, angle_noise_limit)
        cam.rotation_euler.z += random.uniform(-angle_noise_limit, angle_noise_limit)
        
        bpy.context.view_layer.update()
        
        swimmer_mesh = bpy.data.objects.get(swimmer_name)
        rig_name = swimmer_name.replace('.mesh', '')
        swimmer_rig = bpy.data.objects.get(rig_name)

        if not swimmer_mesh or not swimmer_rig:
            cam.rotation_euler = original_rotation
            continue

        hide_all_foreground_swimmers()
        swimmer_mesh.hide_render = False
        swimmer_mesh.hide_viewport = False
        
        swimmer_rig.hide_render = False
        swimmer_rig.hide_viewport = False
        
        set_foreground_materials()

        bg_enabled = random.random() < 0.5
        toggle_background_swimmers(bg_enabled)
        
        if bg_enabled:
            randomize_background_materials()

        swimmer_base_name = swimmer_name.replace('.mesh', '')
        swimmer_folder = os.path.join(out_dir_abs, swimmer_base_name)
        os.makedirs(swimmer_folder, exist_ok=True)

        annotations_data = {}

        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)

            img_filename = f"{swimmer_base_name}_{frame:04d}.png"
            scene.render.filepath = os.path.join(swimmer_folder, img_filename)
            bpy.ops.render.render(write_still=True)

            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_armature = swimmer_rig.evaluated_get(depsgraph)

            frame_kpts = extract_keypoints(scene, cam, eval_armature, res_x, res_y)
            annotations_data[img_filename] = frame_kpts

        json_path = os.path.join(swimmer_folder, "annotations.json")
        with open(json_path, 'w') as f:
            json.dump(annotations_data, f, indent=4)

        cam.rotation_euler = original_rotation
        bpy.context.view_layer.update()

        print(f"{swimmer_name} Finished. Annotations : {json_path}")

if __name__ == "__main__":
    main()
