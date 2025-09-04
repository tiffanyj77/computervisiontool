import cv2
import random
import dlib
import numpy as np

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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces = detector(gray)

for face in faces:
    # Get landmarks
    landmarks = predictor(gray, face)

    # Convert landmarks to (x,y) numpy array
    points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

    # --- Jawline: indices 0–16 ---
    jaw_points = points[0:17]

    # Draw lines between consecutive jaw points
    for i in range(1, len(jaw_points)):
        cv2.line(img,
                 tuple(jaw_points[i-1]),
                 tuple(jaw_points[i]),
                 (0, 255, 0), 2)

    # (Optional) draw all landmark dots
    for (x, y) in points:
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

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


