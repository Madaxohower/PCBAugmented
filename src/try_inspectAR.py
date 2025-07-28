import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

# Camera calibration parameters (replace with your own or use defaults for testing)
camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)

# Parse Eagle .brd file
def load_brd_file(brd_path):
    if not os.path.exists(brd_path):
        print(f"Error: .brd file not found at {brd_path}")
        return []
    try:
        tree = ET.parse(brd_path)
        root = tree.getroot()
        components = []
        for element in root.findall(".//element"):
            name = element.get("name")
            try:
                x = float(element.get("x"))
                y = float(element.get("y"))
                components.append((x, y, name))
            except (TypeError, ValueError):
                print(f"Warning: Skipping component {name} due to invalid x/y coordinates")
        if not components:
            print("Warning: No valid components found in .brd file")
        return components
    except ET.ParseError as e:
        print(f"Error parsing .brd file: {e}")
        return []

# Initialize ORB detector for Arduino board detection
orb = cv2.ORB_create(nfeatures=5000)  # Increased keypoints for better detection
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load reference image of Arduino board
ref_img_path = os.path.join(os.path.dirname(__file__), 'ard.png')
ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
if ref_img is None:
    print(f"Error: Could not load reference image at {ref_img_path}")
    exit()
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
if des_ref is None:
    print("Error: No features detected in reference image")
    exit()

# Load .brd file
brd_path = os.path.join(os.path.dirname(__file__), 'MEGA2560_Rev3e.brd')
components = load_brd_file(brd_path)
if not components:
    print("No components loaded from .brd file. Exiting.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)
    if des_frame is None:
        cv2.imshow('AR PCB Overlay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Match features
    matches = bf.match(des_ref, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    if len(matches) > 10:  # Minimum matches for homography
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Apply RANSAC for homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is not None:
            # Draw board outline
            h, w = ref_img.shape
            corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, homography)
            frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

            # Overlay .brd components
            for comp_x, comp_y, comp_name in components:
                pt = np.float32([[comp_x, comp_y]]).reshape(-1, 1, 2)
                transformed_pt = cv2.perspectiveTransform(pt, homography)
                x, y = int(transformed_pt[0, 0, 0]), int(transformed_pt[0, 0, 1])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(frame, comp_name, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the frame
    cv2.imshow('AR PCB Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()