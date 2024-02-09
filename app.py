import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

def main():
    st.title("Virtual Necklace Try-On App")

    # Load the necklace image
    necklace_image_path = "static/necklace_1.png"
    necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to the index of the desired webcam

    # Create a placeholder for the image
    image_placeholder = st.empty()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Get the width and height of the frame
        height, width, _ = frame.shape

        # Calculate the width of each section
        center_width = int(width * 0.35)
        side_width = (width - center_width) // 2

        # Divide the frame into three sections
        left_section = frame[:, :side_width]
        center_section = frame[:, side_width:side_width + center_width]
        right_section = frame[:, side_width + center_width:]

        # Draw vertical lines to split the screen
        cv2.line(frame, (side_width, 0), (side_width, height), (0, 255, 0), 1)
        cv2.line(frame, (side_width + center_width, 0), (side_width + center_width, height), (0, 255, 0), 1)

        frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
                
        if results.detections:
            for detection in results.detections:
                # Iterate over the landmarks and draw them on the frame
                for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                    # Get the pixel coordinates of the landmark
                    cx, cy = int(landmark.x * center_width), int(landmark.y * height)

                    # Check if hand is within the valid region
                    if cx >= 0 and cx <= center_width:
                        # Extract the bounding box coordinates
                        bboxC = detection.location_data.relative_bounding_box
                        hC, wC, _ = center_section.shape
                        xminC = int(bboxC.xmin * wC)
                        yminC = int(bboxC.ymin * hC)
                        widthC = int(bboxC.width * wC)
                        heightC = int(bboxC.height * hC)
                        xmaxC = xminC + widthC
                        ymaxC = yminC + heightC

                        # Calculate the bottom bounding box coordinates
                        bottom_ymin = ymaxC + 10
                        bottom_ymax = min(ymaxC + 150, hC)

                        # Increase the width of the red bounding box
                        xminC -= 20  # Decrease the left side
                        xmaxC += 20  # Increase the right side

                        # Check if the bounding box dimensions are valid
                        if widthC > 0 and heightC > 0 and xmaxC > xminC and bottom_ymax > bottom_ymin:
                            # Resize necklace image to fit the bounding box size
                            resized_image = cv2.resize(necklace_image, (xmaxC - xminC, bottom_ymax - bottom_ymin))

                            # Calculate the start and end coordinates for the necklace image
                            start_x = xminC
                            start_y = bottom_ymin
                            end_x = start_x + (xmaxC - xminC)
                            end_y = start_y + (bottom_ymax - bottom_ymin)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_image[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the necklace image
                            overlay = resized_image[:, :, :3] * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image
                            region = center_section[start_y:end_y, start_x:end_x]
                            resized_mask_inv = None
                            if region.shape[1] > 0 and region.shape[0] > 0:
                                resized_mask_inv = cv2.resize(mask_inv, (region.shape[1], region.shape[0]))
                                resized_mask_inv = resized_mask_inv[:, :, np.newaxis]  # Add an extra dimension to match the number of channels

                            if resized_mask_inv is not None:
                                region_inv = region * resized_mask_inv

                                # Combine the resized image and the input image region
                                resized_overlay = None
                                if region_inv.shape[1] > 0 and region_inv.shape[0] > 0:
                                    resized_overlay = cv2.resize(overlay, (region_inv.shape[1], region_inv.shape[0]))

                                # Combine the resized overlay and region_inv
                                region_combined = cv2.add(resized_overlay, region_inv)

                                # Replace the neck region in the input image with the combined region
                                center_section[start_y:end_y, start_x:end_x] = region_combined

        # Display the output frame
        image_placeholder.image(frame, channels="BGR")

    # Release the webcam
    cap.release()

if __name__ == "__main__":
    main()
