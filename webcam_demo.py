import torch
import cv2
import time
import argparse
from openai import OpenAI
import numpy as np

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0) # 1
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.1)#0.7125)
args = parser.parse_args()

# client = OpenAI(
#     api_key='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',
#     organization='org-BzkVKW4kSGfQSRNOmP6h1IU2',
# )

# stream = client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "Say this is a test"}],
#     stream=True,
# )

# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")

def calculate_roll_and_approximate_pitch(keypoint_coords):
    # Extract keypoints
    left_eye = np.array(keypoint_coords[1][:2])
    right_eye = np.array(keypoint_coords[2][:2])
    left_shoulder = np.array(keypoint_coords[5][:2])
    right_shoulder = np.array(keypoint_coords[6][:2])

    # Calculate vectors
    eye_vector = right_eye - left_eye
    shoulder_vector = right_shoulder - left_shoulder

    # Roll: angle of the eye line relative to horizontal
    roll = np.arctan2(eye_vector[1], eye_vector[0]) * (180 / np.pi)

    # Normalize roll to [-180, 180] range
    if roll > 180:
        roll -= 360
    roll += 90

    # Pitch approximation: We'll define pitch based on the distance between the eye line and a point in the middle of the shoulder line
    # This is a heuristic and doesn't directly correspond to pitch but can give some indication of head tilt
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    eye_mid = (left_eye + right_eye) / 2
    mid_vector = eye_mid - shoulder_mid
    pitch = np.arctan2(mid_vector[1], mid_vector[0]) * (180 / np.pi) - roll  # Adjusting by roll for a rough approximation

    # Normalize pitch to make it more intuitive (optional)
    pitch = -pitch # Inverting to match conventional pitch directions
    pitch += 180
    if pitch > 360:
        pitch -= 360

    return roll, pitch

def main():
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image)#.cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords, _ = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
            
        for pi, (pose_score, keypoint_score, keypoint_coord) in enumerate(zip(pose_scores, keypoint_scores, keypoint_coords)):
            if pose_score > 0.15:
                roll, pitch = calculate_roll_and_approximate_pitch(keypoint_coord)
                print(f"Pose {pi}: Roll: {roll:.2f} degrees, Approximate Pitch: {pitch:.2f} degrees")

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()