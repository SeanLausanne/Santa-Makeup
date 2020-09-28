import numpy as np
import cv2
import dlib
import time


hat_path = "./materials/christmas_hat.png"
hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
r, g, b, alpha = cv2.split(hat_img)
hat_rgb = cv2.merge((r, g, b))

predictor_path = "./models/shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
RESIZE_RATIO = 2


def put_hat_on_image(img):
    resize_shape = (img.shape[1] // RESIZE_RATIO, img.shape[0] // RESIZE_RATIO)
    # canvas = np.zeros((resize_shape[1], resize_shape[0], 3), dtype=np.uint8)
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

        if target_x < 0 or target_y < 0 or target_x + target_w > img.shape[1] or target_y + target_h > img.shape[0]:
            return img

        # cv2.rectangle(img, (target_x, target_y), (target_x + target_w, target_y + target_h), color=(255, 0, 0), thickness=2, lineType=8)

        # key points
        key_points = predictor(img, target)
        # for point in key_points.parts():
        #     cv2.circle(img, (point.x, point.y), radius=3, color=(0, 255, 0))


        # two eyes
        point1 = key_points.part(0)
        point2 = key_points.part(2)
        eyes_center = ((point1.x + point2.x) // 2, (point1.y + point2.y) // 2)

        # adjust hat size based on the face
        # Eunbi = 2, izone = 1.7
        face_hat_factor = 1.7
        resized_hat_h = int(round(hat_rgb.shape[0] * target_w / hat_rgb.shape[1] * face_hat_factor))
        resized_hat_w = int(round(hat_rgb.shape[1] * target_w / hat_rgb.shape[1] * face_hat_factor))

        if resized_hat_h > target_y:
            resized_hat_h = target_y - 1
            #continue

        # create hat mask
        hat_resized = cv2.resize(hat_rgb, (resized_hat_w, resized_hat_h))
        hat_mask = cv2.resize(alpha, (resized_hat_w, resized_hat_h))
        hat_mask_inv = cv2.bitwise_not(hat_mask)
        # cv2.imshow("hat_mask", hat_mask)
        # cv2.imshow("hat_mask_inv", hat_mask_inv)
        # cv2.waitKey(0)

        # offset between hat and face box
        # Eunbi = 0.7, 0.35, izone = 0.65, 0.2
        offset_ratio_h = 0.5
        offset_ratio_w = 0.2
        offset_h = int(target_h * offset_ratio_h)
        offset_w = int(target_w * offset_ratio_w)

        # ROI in the original img
        roi_y1 = target_y + offset_h - resized_hat_h
        roi_y2 = target_y + offset_h
        roi_x1 = eyes_center[0] + offset_w - resized_hat_w // 2
        roi_x2 = eyes_center[0] + offset_w + resized_hat_w // 2
        bg_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        # cv2.imshow("roi", bg_roi)
        # cv2.waitKey(0)

        hat_mask_inv = cv2.merge((hat_mask_inv, hat_mask_inv, hat_mask_inv))
        hat_mask_inv_copy = hat_mask_inv.copy()
        bg_roi = bg_roi.astype(float)
        hat_mask_roi = hat_mask_inv.astype(float) / 255
        # cv2.imshow("hat_mask_inv", hat_mask_inv)
        # cv2.waitKey(0)

        # put hat mask on ROI
        hat_mask_roi = cv2.resize(hat_mask_roi,
                                  (bg_roi.shape[1], bg_roi.shape[0]))  # in case of size difference due to rounding
        roi_with_mask = cv2.multiply(hat_mask_roi, bg_roi)
        roi_with_mask = roi_with_mask.astype('uint8')
        # cv2.imshow("roi_mask", roi_mask)
        # cv2.waitKey(0)

        # seperate hat from background
        hat_no_bg = cv2.bitwise_and(hat_resized, hat_resized, mask=hat_mask)
        hat_no_bg = cv2.resize(hat_no_bg,
                               (bg_roi.shape[1], bg_roi.shape[0]))  # in case of size difference due to rounding
        # cv2.imshow("hat_no_bg", hat_no_bg)
        # cv2.waitKey(0)

        # put the hat in the original image
        roi_with_hat = cv2.add(roi_with_mask, hat_no_bg)
        img[roi_y1:roi_y2, roi_x1:roi_x2] = roi_with_hat

    return img


video_path = "./materials/izone_vlive.mp4"
cap = cv2.VideoCapture(0)

# _, frame = cap.read()
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# output_video = cv2.VideoWriter('data/izone_vlive_hat.avi', fourcc, 30, (frame.shape[1] // RESIZE_RATIO, frame.shape[0] // RESIZE_RATIO))

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