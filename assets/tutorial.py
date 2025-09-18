import cv2
import random
import dlib
import mediapipe as mp
import numpy as np
from pathlib import Path
import math

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
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 5) #5 refers to the thickness
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


# ---------- I/O ----------
IMG_NAME = "meow.png"
img_path = Path(IMG_NAME)
out_path = img_path.with_name(img_path.stem + "_mixed.png")

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"Couldn't open {img_path.resolve()}")
h, w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Face landmarks ----------
mpfm = mp.solutions.face_mesh
with mpfm.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as fm:
    res = fm.process(rgb)

if not res.multi_face_landmarks:
    print("No face found.")
    raise SystemExit

landmarks = res.multi_face_landmarks[0]
pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark], dtype=np.int32)

# Collect outer-face (oval) points -> clean boundary order via convex hull
oval_idx = sorted({i for a, b in mpfm.FACEMESH_FACE_OVAL for i in (a, b)})
oval_pts = pts[oval_idx]
hull = cv2.convexHull(oval_pts)[:, 0, :]   # (N, 2), ordered around contour


# If you already defined arc_mean_y earlier, you can skip redefining it.
def arc_mean_y(center, axes, angle_deg, start_deg, end_deg, n=60):
    cx, cy = center
    rx, ry = axes
    theta = np.deg2rad(np.linspace(start_deg, end_deg, n))
    x = rx * np.cos(theta)
    y = ry * np.sin(theta)
    a = math.radians(angle_deg)
    xr =  x * math.cos(a) - y * math.sin(a)
    yr =  x * math.sin(a) + y * math.cos(a)
    y_img = cy + yr
    return float(np.mean(y_img))

def ellipse_point(center, axes, angle_deg, t_deg):
    cx, cy = center
    rx, ry = axes
    t = math.radians(t_deg)
    a = math.radians(angle_deg)
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    xr =  x * math.cos(a) - y * math.sin(a)
    yr =  x * math.sin(a) + y * math.cos(a)
    return (int(round(cx + xr)), int(round(cy + yr)))

import math

def ellipse_tangent(center, axes, angle_deg, t_deg):
    """Unit tangent on a rotated ellipse (image coords)."""
    cx, cy = center
    rx, ry = axes
    t = math.radians(t_deg)
    a = math.radians(angle_deg)
    dx = -rx * math.sin(t)
    dy =  ry * math.cos(t)
    tx = dx * math.cos(a) - dy * math.sin(a)
    ty = dx * math.sin(a) + dy * math.cos(a)
    n = math.hypot(tx, ty) + 1e-8
    return (tx / n, ty / n)

def cubic_bezier(p0, p1, p2, p3, n=160):
    """Sample cubic Bézier p0..p3."""
    import numpy as np
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    p3 = np.array(p3, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=True).reshape(-1, 1)
    pts = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    return pts.astype(np.int32)


# ---------- Top ellipse (forehead/head) with robust half selection ----------
if len(hull) >= 5:
    (cx, cy), (maj, minr), angle = cv2.fitEllipse(hull.astype(np.float32))
    center = (int(cx), int(cy))
    # fitEllipse gives diameters; cv2.ellipse expects radii
    axes = (int(maj / 2), int(minr / 2))

    # Decide which half is the true "top" (smaller average y)
    mean_top = arc_mean_y(center, axes, angle, 0, 180)
    mean_bottom = arc_mean_y(center, axes, angle, 180, 360)
    if mean_top < mean_bottom:
        start_deg, end_deg = 0, 180
    else:
        start_deg, end_deg = 180, 360

    # Sample arc endpoints; nudge inward a bit to avoid overhangs
    p_start = ellipse_point(center, axes, angle, start_deg + 2)
    p_end   = ellipse_point(center, axes, angle, end_deg - 2)
else:
    # Fallback if ellipse can't be fitted
    center = (int(np.mean(hull[:, 0])), int(np.median(hull[:, 1])))
    p_start = tuple(hull[np.argmin(hull[:, 0])])
    p_end   = tuple(hull[np.argmax(hull[:, 0])])

# ---------- Build one continuous CLOSED outline: top ellipse arc + jaw (reversed) ----------
jaw = hull[hull[:, 1] >= center[1]]

if len(jaw) >= 2:
    # order ellipse endpoints left→right and match their degrees
    left_ep, right_ep = (p_start, p_end) if p_start[0] <= p_end[0] else (p_end, p_start)
    left_deg, right_deg = (start_deg + 2, end_deg - 2) if left_ep == p_start else (end_deg - 2, start_deg + 2)

    # ellipse tangents at the temples
    tL_ell = ellipse_tangent(center, axes, angle, left_deg)
    tR_ell = ellipse_tangent(center, axes, angle, right_deg)

    # jaw polyline left→right and local tangents near temples
    jaw_lr = jaw[np.argsort(jaw[:, 0])]
    if len(jaw_lr) < 3:
        jaw_lr = jaw
    def tangent_at(poly, i):
        if i <= 0:
            v = poly[1] - poly[0]
        elif i >= len(poly)-1:
            v = poly[-1] - poly[-2]
        else:
            v = poly[i+1] - poly[i-1]
        n = np.linalg.norm(v) + 1e-8
        return (v[0]/n, v[1]/n)
    iL = np.argmin(np.sum((jaw_lr - np.array(left_ep))**2, axis=1))
    iR = np.argmin(np.sum((jaw_lr - np.array(right_ep))**2, axis=1))
    tL_jaw = tangent_at(jaw_lr, iL)
    tR_jaw = tangent_at(jaw_lr, iR)

    # blend directions to avoid a flat top
    # chord direction left→right
    u = np.array([right_ep[0] - left_ep[0], right_ep[1] - left_ep[1]], dtype=np.float32)
    u /= (np.linalg.norm(u) + 1e-8)

    ALPHA_ELL = 0.65   # weight for ellipse tangent (0.5–0.8)
    MIX_JAW   = 0.35   # how much jaw tangent influences (0–0.5)

    dL = ALPHA_ELL*np.array(tL_ell) + (1-ALPHA_ELL)*u
    dL /= (np.linalg.norm(dL) + 1e-8)
    dR = ALPHA_ELL*(-np.array(tR_ell)) + (1-ALPHA_ELL)*(-u)
    dR /= (np.linalg.norm(dR) + 1e-8)

    # blend in jaw tangents
    dL = (1-MIX_JAW)*dL + MIX_JAW*np.array(tL_jaw); dL /= (np.linalg.norm(dL)+1e-8)
    dR = (1-MIX_JAW)*dR - MIX_JAW*np.array(tR_jaw); dR /= (np.linalg.norm(dR)+1e-8)

    # control point distances scale with temple angle + chord length
    def angle_between(a, b):
        a = a/(np.linalg.norm(a)+1e-8); b = b/(np.linalg.norm(b)+1e-8)
        return float(np.degrees(np.arccos(np.clip(np.dot(a,b), -1.0, 1.0))))
    chord = float(np.linalg.norm(np.array(right_ep) - np.array(left_ep)))
    angL = angle_between(dL, u);   angR = angle_between(-dR, -u)
    base = 0.40 * chord   # baseline curvature (raise to round more)
    gain = 0.004          # extra curvature per degree (0.003–0.007)
    sL = base + gain * angL * chord
    sR = base + gain * angR * chord

    # cubic Bézier control points along the blended directions
    p0 = left_ep
    p1 = (int(round(p0[0] + sL * dL[0])), int(round(p0[1] + sL * dL[1])))
    p3 = right_ep
    p2 = (int(round(p3[0] - sR * dR[0])), int(round(p3[1] - sR * dR[1])))

    # forehead bridge sampled smoothly
    bridge_pts = cubic_bezier(p0, p1, p2, p3, n=160)

    # light smoothing for jaw, then build closed loop: bridge L→R + jaw R→L
    jaw_lr = cv2.approxPolyDP(jaw_lr.reshape(-1,1,2), epsilon=2.0, closed=False).reshape(-1,2)
    path = np.vstack([bridge_pts, jaw_lr[::-1]]).astype(np.int32)
    cv2.polylines(img, [path], isClosed=True, color=(0,255,0), thickness=2)

else:
    # fallback if jaw not available
    cv2.ellipse(img, center, axes, float(angle), 0, 360, (0,255,0), 2)

# ---- save and show ----

cv2.imshow('Image', img)
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


