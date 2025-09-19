import cv2
import random
import dlib
import mediapipe as mp
import numpy as np
from pathlib import Path
import math
import sys

img = cv2.imread('meow.png', 1)
#alpha channel corresponds to -1 (extra fourth channel for transparency)
cv2.line(img, ((img.shape[1])//2, 0), ((img.shape[1])//2, img.shape[0]), (255, 0, 0), 2)
cv2.line(img, (0, (img.shape[0])//2), (img.shape[1], (img.shape[0])//2), (255, 0, 0), 2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)
#smaller scale factor means that it will detect more faces but it runs slower, larger scale factor means that it will detect less faces but runs faster

for (x, y, w, h) in faces: #x = number of cols, just like in math
    #cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 5) #5 refers to the thickness
    roi_gray = gray[y:y+h, x:x+w] #getting location of face
    roi_color = img[y:y+h, x:x+w] #this is the colored face
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5) #detects eyes in the given gray image (so it doesn't scan the entire face)
    #good practice to upload grayscale image instead of colored image
    eye_centers = []
    for (ex, ey, ew, eh) in eyes:
        cx = ex + ew // 2
        cy = ey + eh // 2
        eye_centers.append((cx, cy))

    # need at least two eyes; pick the two leftmost to avoid eyebrow/noise
    if len(eye_centers) >= 2:
        eye_centers = sorted(eye_centers, key=lambda p: p[0])[:2]
        # average the vertical positions so the line goes through both
        eye_line_y = int((eye_centers[0][1] + eye_centers[1][1]) / 2)

        # draw the guideline across the entire FACE (left to right of the face box)
        cv2.line(roi_color, (0, eye_line_y), (w, eye_line_y), (255, 0, 0), 2)
        #roi_color is the face box


# ------------- Config / I/O -------------
IMG_NAME = sys.argv[1] if len(sys.argv) > 1 else "meow.png"
img_path = Path(IMG_NAME)
out_path = img_path.with_name(img_path.stem + "_outlined.png")

img = cv2.imread(str(img_path), 1)
if img is None:
    raise FileNotFoundError(f"Couldn't open {img_path.resolve()}")

h, w = img.shape[:2]

# ------------- Guide axes (optional) -------------
cv2.line(img, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)
cv2.line(img, (0, h // 2), (w, h // 2), (255, 0, 0), 2)

# ------------- Eye guideline via Haar cascades (optional) -------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

for (x, y, fw, fh) in faces:
    # draw face box just for visualization
    #cv2.rectangle(img, (x, y), (x + fw, y + fh), (255, 0, 0), 2)

    # detect eyes inside the face ROI (grayscale)
    roi_gray  = gray[y:y + fh, x:x + fw]
    roi_color = img[y:y + fh, x:x + fw]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
    eye_centers = []
    for (ex, ey, ew, eh) in eyes:
        cx = ex + ew // 2
        cy = ey + eh // 2
        eye_centers.append((cx, cy))

    # need at least two eyes; take the two leftmost to dodge eyebrow/noise
    if len(eye_centers) >= 2:
        eye_centers = sorted(eye_centers, key=lambda p: p[0])[:2]
        eye_line_y = int((eye_centers[0][1] + eye_centers[1][1]) / 2)
        # draw across the entire FACE box
        cv2.line(roi_color, (0, eye_line_y), (fw, eye_line_y), (255, 0, 0), 2)

# ------------- True face outline with MediaPipe FaceMesh -------------
mpfm = mp.solutions.face_mesh

def face_oval_points_in_order(landmarks, w, h, mpfm):
    """
    Returns the FACEMESH_FACE_OVAL points in correct drawing order (closed loop).
    MediaPipe provides the oval as undirected edges; we stitch them into a cycle.
    """
    # Build adjacency from edges
    edges = list(mpfm.FACEMESH_FACE_OVAL)  # list of (a, b) pairs
    nbrs = {}
    idx_set = set()
    for a, b in edges:
        idx_set.add(a); idx_set.add(b)
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)

    # Convert ALL mediapipe landmarks to pixel coords once
    all_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

    # Start at leftmost oval vertex (smallest x)
    leftmost_idx = min(idx_set, key=lambda i: all_pts[i][0])

    # Walk the cycle (each oval vertex should have degree 2)
    order = [leftmost_idx]
    prev = None
    cur = leftmost_idx
    # guard to avoid infinite loops if something odd happens
    for _ in range(len(idx_set) + 5):
        neighbors = nbrs[cur]
        nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
        if nxt == order[0]:  # closed loop
            break
        order.append(nxt)
        prev, cur = cur, nxt

    # Map ordered indices to pixel coords
    oval_pts = np.array([all_pts[i] for i in order], dtype=np.int32)
    return oval_pts

# Run FaceMesh once on the (BGR->)RGB image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with mpfm.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True  # helps around contours/eyes
) as fm:
    res = fm.process(rgb)

if not res.multi_face_landmarks:
    print("No face found by MediaPipe FaceMesh.")
else:
    landmarks = res.multi_face_landmarks[0]
    oval_pts = face_oval_points_in_order(landmarks, w, h, mpfm)

    # Draw closed outline = jawline + forehead curve
    cv2.polylines(img, [oval_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # If you want ONLY the jawline, you could do:
    # mid_y = np.median(oval_pts[:, 1])
    # jaw_pts = oval_pts[oval_pts[:, 1] >= mid_y]
    # cv2.polylines(img, [jaw_pts], isClosed=False, color=(0, 255, 0), thickness=2)

# ------------- Save & show -------------
cv2.imwrite(str(out_path), img)
print(f"Saved: {out_path}")

cv2.imshow("Face Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#width x height REMEMEBER
#number of columns x number of rows

# for i in range(100): #(number of rows) aka height
#     for j in range(img.shape[1]): #(number of columns) aka width
#         img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] #so this works with three channels now that it's not transparent anymore
#coordinate system
# #copying an image by copying parts of an array
# tag = img[500:700, 600:900]
# img[100:300, 650:950] = tag
# cv2.imshow('Image', img)
# for i in range(100): (number of rows) aka height
#     for j in range(img.shape[1]): (number of columns) aka width
#         img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
#         #each place in the array will have a random pixel
# shape is an array, shape[] = [height, width, channels]
#shape is referring to the dimensions of the entire image
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 1 = color, neglect transparency 0 = grayscale, -1 = image as it is, include transparency
# img = cv2.resize(img, (0,0), fx = 2, fy = 2) #can resize it by pixels or image ratio
# #you are in the assets folder so you don't need /assets
# #.5 halves the image
# img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
# cv2.imwrite('new_img.jpg', img)
# #saves the image as a new image in the assets folder

# cv2.imshow('Image', img) #display the image
# cv2.waitKey(0) #wait an infinite amount of time = 0, if put 5 then wait up to 5 seconds for key to be pressed
# cv2.destroyAllWindows()

#channel is a how many values are representing each pixel (in opencv it is blue green red)
#0-255 is your limit for bgr colors
#[


