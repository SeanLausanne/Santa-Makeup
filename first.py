import cv2
import dlib
import time


predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

start = time.perf_counter()
idx = 1
while True:
    ret, frame = cap.read()
    dets = detector(frame, 0)

    if len(dets) > 0:
        for d in dets:
            x, y, w, h = d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2, lineType=8)
            shape = predictor(frame, d)
            for point in shape.parts():
                cv2.circle(frame, (point.x, point.y), radius=4, color=(0, 255, 0), thickness=1)

    cv2.imshow("frame", frame)
    print("frame", idx)
    idx += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end = time.perf_counter()
length = end - start
print("time:", length)
print("fps", idx / length)

cap.release()
cv2.destroyAllWindows()
