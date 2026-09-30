import cv2
import json
import os
import numpy as np

RENDER_DIR = "results/render"
OUTPUT_DIR = "results/validation"

POINT_RADIUS = 2
OUTLINE_RADIUS = 3
LEGEND_WIDTH = 180
BG_COLOR = (40, 40, 40)

COLORS = {
    'head': (0, 0, 255),
    'sternum': (0, 128, 255),
    'left_shoulder': (0, 255, 255),
    'right_shoulder': (0, 200, 200),
    'left_elbow': (0, 255, 0),
    'right_elbow': (0, 128, 0),
    'left_hand': (255, 0, 0),
    'right_hand': (128, 0, 0),
    'left_hip': (255, 0, 255),
    'right_hip': (128, 0, 128),
    'left_knee': (255, 255, 0),
    'right_knee': (128, 128, 0),
    'left_foot': (255, 255, 255),
    'right_foot': (150, 150, 150)
}
# =================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for swimmer_dir in os.listdir(RENDER_DIR):
        swimmer_path = os.path.join(RENDER_DIR, swimmer_dir)
        
        if not os.path.isdir(swimmer_path):
            continue

        json_path = os.path.join(swimmer_path, "annotations.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        out_swimmer_dir = os.path.join(OUTPUT_DIR, swimmer_dir)
        os.makedirs(out_swimmer_dir, exist_ok=True)

        for img_name, keypoints in annotations.items():
            img_path = os.path.join(swimmer_path, img_name)

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, c = img.shape
            line_height = 20
            required_height = 30 + len(COLORS) * line_height

            new_h = max(h, required_height)
            new_w = w + LEGEND_WIDTH

            canvas = np.zeros((new_h, new_w, c), dtype=np.uint8)
            canvas[:] = BG_COLOR
            canvas[0:h, 0:w] = img

            legend_start_x = w + 10
            legend_start_y = 25

            for i, (kp_name, color) in enumerate(COLORS.items()):
                y_legend = legend_start_y + i * line_height
                cv2.circle(canvas, (legend_start_x + 10, y_legend - 4), radius=4, color=color, thickness=-1)
                cv2.putText(canvas, kp_name, (legend_start_x + 25, y_legend),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            for kp_name, kp_data in keypoints.items():
                x, y = int(kp_data['x']), int(kp_data['y'])
                is_visible = kp_data['in_front_of_cam']
                color = COLORS.get(kp_name, (255, 255, 255))

                if is_visible and (0 <= x < w) and (0 <= y < h):
                    cv2.circle(canvas, (x, y), radius=POINT_RADIUS, color=color, thickness=-1)
                    cv2.circle(canvas, (x, y), radius=OUTLINE_RADIUS, color=(0, 0, 0), thickness=1)

            out_path = os.path.join(out_swimmer_dir, img_name)
            cv2.imwrite(out_path, canvas)

    print(f"\nRendered in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
