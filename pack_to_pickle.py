import os
import json
import pickle
import numpy as np
import cv2

RENDER_DIR = "results/render"
OUTPUT_DIR = "results/synth"

STRIDE = 7

KP_ORDER = [
    'head', 'sternum', 'left_shoulder', 'left_elbow', 'left_hand',
    'right_shoulder', 'right_elbow', 'right_hand', 'left_hip',
    'left_knee', 'left_foot', 'right_hip', 'right_knee', 'right_foot'
]

# Stack temporal window [t-3 to t+3] into 21 channels (7 frames x 3 RGB channels)
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    packed_count = 0

    for swimmer_dir in os.listdir(RENDER_DIR):
        swimmer_path = os.path.join(RENDER_DIR, swimmer_dir)
        
        if not os.path.isdir(swimmer_path):
            continue

        json_path = os.path.join(swimmer_path, "annotations.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        frames_dispos = sorted(annotations.keys())
        
        if len(frames_dispos) < 7:
            continue

        base_name = frames_dispos[0].rsplit('_', 1)[0]
        frames_int = sorted([int(f.rsplit('_', 1)[1].replace('.png', '')) for f in frames_dispos])

        print(f"[{swimmer_dir}] - {len(frames_int)} frames found.")

        start_t = frames_int[0] + 3
        end_t = frames_int[-1] - 3

        for t_int in range(start_t, end_t + 1, STRIDE):
            seq_names = [f"{base_name}_{f:04d}.png" for f in range(t_int - 3, t_int + 4)]

            if not all(f in annotations for f in seq_names):
                continue

            center_frame_name = seq_names[3]
            
            img_list = []
            for f_name in seq_names:
                img_path = os.path.join(swimmer_path, f_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_list.append(img)

            stack_7 = np.stack(img_list, axis=2)

            H, W, _, _ = stack_7.shape
            images_21 = stack_7.reshape((H, W, 21))

            center_kps_data = annotations[center_frame_name]
            kp_2d = np.zeros((14, 2), dtype=np.float32)

            for kp_idx, kp_name in enumerate(KP_ORDER):
                if kp_name in center_kps_data:
                    kp_info = center_kps_data[kp_name]
                    x, y = float(kp_info['x']), float(kp_info['y'])

                    if kp_info['in_front_of_cam'] and (0 <= x < W) and (0 <= y < H):
                        kp_2d[kp_idx, 0] = x
                        kp_2d[kp_idx, 1] = y
                    else:
                        kp_2d[kp_idx, :] = np.nan
                else:
                    kp_2d[kp_idx, :] = np.nan

            out_filename = f"synth_{center_frame_name.replace('.png', '.p')}"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            data = {
                "images": images_21,
                "body_markers": kp_2d
            }

            with open(out_path, 'wb') as f:
                pickle.dump(data, f)

            packed_count += 1

    print(f"\n{packed_count} sequences (.p) generated in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
