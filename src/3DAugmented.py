import argparse
import cv2
import numpy as np
import os
from obj_loader import *

# ARUCO DICTIONARY AND PARAMETERS
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# CAMERA PARAMETERS RESOLUTION 800X600
CAMERA_MATRIX = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

# MARKER SIZE
MARKER_SIZE = 0.01  # 0.1 = 10 cm

# MAIN FUNCTION THAT DETECT 3D MODEL
def main():

    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, '..', 'models', 'pirate-ship-fat.obj'), swapyz=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to capture video")
        return

    # CAMERA RESOLUTION
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    #RESOLUTION VERIFICATION
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {width}x{height}")

    #UI WINDOW SIZE
    cv2.namedWindow('Augmented Reality', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Augmented Reality', 800, 600)

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    color = parse_color(args.color)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            if args.draw_marker:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)
            for i in range(len(ids)):
                if args.draw_axis:
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], MARKER_SIZE / 2)
                projection = projection_matrix(rvecs[i], tvecs[i])
                try:
                    frame = render(frame, obj, projection, MARKER_SIZE, args.object_scale, color=color)
                except Exception as e:
                    print(f"Rendering error: {e}")
        cv2.imshow('Augmented Reality', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# RENDER IMG, OBJ, PROJECTION, MARKER_SIZE, OBJECT_SCALE, COLOR
def render(img, obj, projection, marker_size, object_scale=0.01, color=(0, 0, 0)):

    vertices = obj.vertices
    scale_matrix = np.eye(3) * (marker_size * object_scale)
    marker_center = marker_size / 2
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + marker_center, p[1] + marker_center, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is None:
            try:
                color_val = hex_to_rgb(face[-1], default_color=(0, 0, 255))
                color_val = color_val[::-1]
            except Exception as e:
                print(f"Color error for face {face}: {e}")
                color_val = (0, 0, 255)
        else:
            color_val = color[::-1]
        cv2.fillConvexPoly(img, imgpts, color_val)
    return img

# CREATE A 3D PROJECTION MATRIX USING ROTATION AND TRANSLATION VECTORS
def projection_matrix(rvec, tvec):

    rmat, _ = cv2.Rodrigues(rvec)
    projection = np.hstack((rmat, tvec.reshape(-1, 1)))
    projection = np.dot(CAMERA_MATRIX, projection)
    return projection

# CONVERT HEX COLOR STRING TO RGB TUPLE
def hex_to_rgb(hex_color, default_color=(0, 0, 255)):

    try:
        if not isinstance(hex_color, str):
            print(f"Warning: Invalid hex_color type: {type(hex_color)}, value: {hex_color}")
            return default_color
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        if h_len not in (3, 6):
            print(f"Warning: Invalid hex_color length: {hex_color}")
            return default_color
        return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    except (ValueError, TypeError) as e:
        print(f"Error parsing hex_color '{hex_color}': {e}")
        return default_color

# PARSE COLOR STRING TO RGB TUPLE
def parse_color(color_str):

    colors = {
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'white': (255, 255, 255),
        'medium_gray': (128, 128, 128)
    }
    if color_str.lower() in colors:
        return colors[color_str.lower()]
    if color_str.startswith('#'):
        return hex_to_rgb(color_str)
    return (0, 0, 0)  # Default to black


#COMMAND-LINE ARGUMENT
parser = argparse.ArgumentParser(description='Augmented reality using ArUco markers')
parser.add_argument('--draw-marker', action='store_true', help='Draw detected ArUco marker borders')
parser.add_argument('--draw-axis', action='store_true', help='Draw coordinate axes on detected markers')
parser.add_argument('--object-scale', type=float, default=1.5,
                    help='Scaling factor for the 3D object (default: 0.005)')
parser.add_argument('--color', type=str, default='red', help='Object color (e.g., red, blue, #FF0000, or black)')
args = parser.parse_args()

if __name__ == '__main__':
    main()