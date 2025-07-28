import cv2
import numpy as np
import os
from lxml import etree

# Camera calibration parameters (replace with calibrated values)
camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)

# Parse Eagle .brd file with wires, components, and packages
def parse_brd_file(brd_path):
    try:
        tree = etree.parse(brd_path)
        root = tree.getroot()
        wires = []
        components = []
        packages = {}

        for package in root.findall(".//package"):
            name = package.get('name')
            outlines = []
            for wire in package.findall(".//wire"):
                try:
                    x1 = float(wire.get('x1', 0))
                    y1 = float(wire.get('y1', 0))
                    x2 = float(wire.get('x2', 0))
                    y2 = float(wire.get('y2', 0))
                    width = float(wire.get('width', 0.1))
                    outlines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'width': width})
                except:
                    continue
            if outlines:
                packages[name] = outlines

        for wire in root.findall(".//wire"):
            try:
                x1 = float(wire.get('x1', 0))
                y1 = float(wire.get('y1', 0))
                x2 = float(wire.get('x2', 0))
                y2 = float(wire.get('y2', 0))
                layer = wire.get('layer', '1')
                width = float(wire.get('width', 0.1))
                wires.append({
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'layer': layer,
                    'width': width,
                    'visible': True
                })
            except:
                continue

        for element in root.findall(".//element"):
            name = element.get('name', 'Unknown')
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            rot = element.get('rot', '')
            layer = element.get('layer', '1')
            package_name = element.get('package', '')
            components.append({
                'name': name,
                'x': x,
                'y': y,
                'rot': rot,
                'layer': layer,
                'package': package_name,
                'outline': packages.get(package_name, [])
            })

        # Initial scale based on maximum dimensions (will be adjusted dynamically)
        x_coords = [x for x in [w['x1'] for w in wires] + [w['x2'] for w in wires] + [c['x'] for c in components]]
        y_coords = [y for y in [w['y1'] for w in wires] + [w['y2'] for w in wires] + [c['y'] for c in components]]
        board_width = max(x_coords) - min(x_coords) if x_coords else 1
        board_height = max(y_coords) - min(y_coords) if y_coords else 1
        initial_scale = max(board_width, board_height)  # Use raw max dimensions for dynamic scaling

        return wires, components, initial_scale
    except Exception as e:
        print(f"Error parsing BRD file: {e}")
        return [], [], 1.0

# Point-line distance function
def point_line_distance(pt, line_start, line_end):
    x, y = pt
    x1, y1 = line_start
    x2, y2 = line_end
    if (x1, y1) == (x2, y2):
        return np.hypot(x - x1, y - y1)
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    return np.hypot(x - xx, y - yy)

# Mouse click callback
def on_mouse_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    wires, scale = param['wires'], param['scale']
    buttons = param['buttons']
    clicked = False

    for button in buttons:
        if button['x'] <= x <= button['x'] + button['w'] and button['y'] <= y <= button['y'] + button['h']:
            button['state'] = not button['state']
            for wire in wires:
                if wire['layer'] == button['layer']:
                    wire['visible'] = button['state']
            param['update'] = True
            return

    for wire in wires:
        pt1 = (int(wire['x1'] * scale), int(wire['y1'] * scale))
        pt2 = (int(wire['x2'] * scale), int(wire['y2'] * scale))
        dist = point_line_distance((x, y), pt1, pt2)
        if dist < 10:
            wire['visible'] = not wire['visible']
            clicked = True
            break

    if clicked:
        param['update'] = True

# Initialize ORB detector for Arduino board detection
orb = cv2.ORB_create(nfeatures=1500)
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

# Load .brd file and get initial scaling factor
brd_path = os.path.join(os.path.dirname(__file__), 'MEGA2560_Rev3e.brd')
wires, components, initial_scale = parse_brd_file(brd_path)
if not wires:
    print("No wires found or error parsing BRD file. Exiting.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cv2.namedWindow('AR PCB Overlay')
layers = ['1', '16', '2', '15', '3', '14']
buttons = []
for i, layer in enumerate(layers):
    buttons.append({
        'x': 10,
        'y': 10 + i * 50,
        'w': 160,
        'h': 40,
        'state': True,
        'label': f'Toggle Layer {layer}',
        'layer': layer
    })

callback_param = {'wires': wires, 'scale': 6, 'update': True, 'buttons': buttons}
cv2.setMouseCallback('AR PCB Overlay', on_mouse_click, param=callback_param)

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
    if len(matches) > 15:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Apply RANSAC with tight threshold
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if homography is not None:
            # Get reference image corners
            h_ref, w_ref = ref_img.shape
            corners_ref = np.float32([[0, 0], [0, h_ref-1], [w_ref-1, h_ref-1], [w_ref-1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners_ref, homography)
            frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

            # Calculate detected board dimensions in live feed
            top_left = transformed_corners[0][0]
            top_right = transformed_corners[1][0]
            bottom_left = transformed_corners[3][0]
            detected_width = np.hypot(top_right[0] - top_left[0], top_right[1] - top_left[1])
            detected_height = np.hypot(bottom_left[0] - top_left[0], bottom_left[1] - top_left[1])

            # Calculate dynamic scale based on detected dimensions and initial .brd scale
            dynamic_scale = max(detected_width / initial_scale, detected_height / initial_scale)

            # Check orientation and flip if needed
            corner_diff = transformed_corners[1][0][1] - transformed_corners[0][0][1]
            flip_matrix = np.array([[1, 0, 0], [0, -1, h_ref], [0, 0, 1]]) if corner_diff < 0 else np.eye(3)
            homography = homography @ flip_matrix

            # Render wires with layer toggling
            for wire in wires:
                if not wire['visible']:
                    continue
                start_pt = np.float32([[wire['x1'] / dynamic_scale, wire['y1'] / dynamic_scale]]).reshape(-1, 1, 2)
                end_pt = np.float32([[wire['x2'] / dynamic_scale, wire['y2'] / dynamic_scale]]).reshape(-1, 1, 2)
                transformed_start = cv2.perspectiveTransform(start_pt, homography)
                transformed_end = cv2.perspectiveTransform(end_pt, homography)
                x1, y1 = int(transformed_start[0, 0, 0]), int(transformed_start[0, 0, 1])
                x2, y2 = int(transformed_end[0, 0, 0]), int(transformed_end[0, 0, 1])
                color = (255, 0, 0) if wire['layer'] == '1' else (0, 0, 255)
                thickness = max(1, int(wire['width'] * callback_param['scale']))
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

            # Render component outlines
            for comp in components:
                cx, cy = comp['x'] / dynamic_scale, comp['y'] / dynamic_scale
                angle = 0
                if comp['rot'].startswith('R'):
                    try:
                        angle = float(comp['rot'][1:])
                    except:
                        pass
                rad = np.deg2rad(angle)
                cos_r, sin_r = np.cos(rad), np.sin(rad)

                for outline in comp['outline']:
                    dx1, dy1 = outline['x1'], outline['y1']
                    dx2, dy2 = outline['x2'], outline['y2']
                    x1r = cos_r * dx1 - sin_r * dy1 + cx
                    y1r = sin_r * dx1 + cos_r * dy1 + cy
                    x2r = cos_r * dx2 - sin_r * dy2 + cx
                    y2r = sin_r * dx2 + cos_r * dy2 + cy
                    pt1 = np.float32([[x1r, y1r]]).reshape(-1, 1, 2)
                    pt2 = np.float32([[x2r, y2r]]).reshape(-1, 1, 2)
                    transformed_pt1 = cv2.perspectiveTransform(pt1, homography)
                    transformed_pt2 = cv2.perspectiveTransform(pt2, homography)
                    x1, y1 = int(transformed_pt1[0, 0, 0]), int(transformed_pt1[0, 0, 1])
                    x2, y2 = int(transformed_pt2[0, 0, 0]), int(transformed_pt2[0, 0, 1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 150, 0), 1)

            # Draw buttons for layer toggling
            for button in buttons:
                color = (50, 50, 50) if button['state'] else (120, 120, 120)
                cv2.rectangle(frame, (button['x'], button['y']),
                              (button['x'] + button['w'], button['y'] + button['h']),
                              color, -1)
                cv2.putText(frame, button['label'], (button['x'] + 5, button['y'] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('AR PCB Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()