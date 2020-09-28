import numpy as np
import cv2
import dlib
import time


predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
RESIZE_RATIO = 2


def put_hat_on_image(img):

    resize_shape = (img.shape[1] // RESIZE_RATIO, img.shape[0] // RESIZE_RATIO)
    canvas = np.zeros((resize_shape[1], resize_shape[0], 3), dtype=np.uint8)
    img = cv2.resize(img, resize_shape)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # detections, scores, cat = detector.run(img, 1, 0)  # para1: upsampling scale, para2: score threshold
    detections = detector(img_gray, 0)

    for i in range(len(detections)):
        target = detections[i]
        target_x, target_y = target.left(), target.top()
        target_w, target_h = target.right() - target.left(), target.bottom() - target.top()

        # if target_x < 0 or target_y < 0 or target_x + target_w > img.shape[1] or target_y + target_h > img.shape[0]:
        #     continue

        # cv2.rectangle(canvas, (target_x, target_y), (target_x + target_w, target_y + target_h), color=(255, 0, 0),
        #               thickness=2, lineType=8)

        # key points
        key_points = predictor(img, target)
        points = key_points.parts()
        points = np.asarray([(p.x, p.y) for p in points])
        face_contour = points[:17]
        left_eyebrow, right_eyebrow = points[17:22], points[22:27]
        nose_upper, nose_bottom = points[27:31], points[31:36]
        left_eye, right_eye = points[36:42], points[42:48]
        mouth = points[48:68]
        mouth_outer, mouth_inner = points[48:60], points[60:68]

        cv2.polylines(canvas, [face_contour], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, isClosed=False)
        cv2.polylines(canvas, [left_eyebrow], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, isClosed=False)
        cv2.polylines(canvas, [right_eyebrow], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, isClosed=False)
        cv2.polylines(canvas, [nose_upper], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, isClosed=False)
        cv2.polylines(canvas, [nose_bottom], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, isClosed=False)
        cv2.fillPoly(canvas, [left_eye], color=(0, 0, 255), lineType=cv2.LINE_AA)
        cv2.fillPoly(canvas, [right_eye], color=(0, 0, 255), lineType=cv2.LINE_AA)
        cv2.fillPoly(canvas, [mouth], color=(0, 0, 255), lineType=cv2.LINE_AA)

        # for point in mouth_outer:
        #     cv2.circle(canvas, point, radius=3, color=(0, 255, 0))

    canvas = cv2.flip(canvas, 1)
    return canvas

video_path = "./materials/izone_vlive.mp4"

cap = cv2.VideoCapture(0)
start = time.perf_counter()
idx = 1

while True:
    _, frame = cap.read()
    frame_det = put_hat_on_image(frame)
    cv2.imshow("frame", frame_det)
    if idx % 10 == 0:
        print("frame", idx)
    idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.perf_counter()
length = end - start
print("time:", length)
print("fps", idx / length)

cap.release()
cv2.destroyAllWindows()