import numpy as np
import cv2
import dlib
import time


def trim_png_background(img):
    h = img.shape[0]
    w = img.shape[1]
    top_y, bottom_y = 0, h
    left_x, right_x = 0, w

    val = 255* 5
    last_row_sum = 0
    for i in range(h):
        row_sum = sum(img[i, :])
        if row_sum > val:
            if top_y == 0:
                top_y = i
        if row_sum <= val < last_row_sum:
            bottom_y = i - 1
            break
        last_row_sum = row_sum

    last_col_sum = 0
    for i in range(w):
        col_sum = sum(img[:, i])
        if col_sum > val:
            if left_x == 0:
                left_x = i
        if col_sum <= val < last_col_sum:
            right_x = i - 1
            break
        last_col_sum = col_sum

    return top_y, bottom_y, left_x, right_x


hat_path = "./materials/christmas_hat.png"
hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
r, g, b, alpha = cv2.split(hat_img)
hat_rgb = cv2.merge((r, g, b))

top_y, bottom_y, left_x, right_x = trim_png_background(alpha)
alpha = alpha[top_y:bottom_y, left_x:right_x]
hat_rgb = hat_rgb[top_y:bottom_y, left_x:right_x]
# cv2.imshow("hat_rgb", hat_rgb)
# cv2.imshow("alpha", alpha)
# cv2.waitKey(0)


predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
RESIZE_RATIO = 2


def put_hat_on_image(img):
    img = cv2.flip(img, 1)
    resize_shape = (int(img.shape[1] / RESIZE_RATIO), int(img.shape[0] / RESIZE_RATIO))
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
        points = key_points.parts()
        # for point in points:
        #     cv2.circle(img, (point.x, point.y), radius=3, color=(0, 255, 0))

        points = np.asarray([(p.x, p.y) for p in points])
        face_contour = points[:17]
        left_eyebrow, right_eyebrow = points[17:22], points[22:27]
        nose_upper, nose_bottom = points[27:31], points[31:36]
        left_eye, right_eye = points[36:42], points[42:48]
        mouth = points[48:68]
        mouth_outer, mouth_inner = points[48:60], points[60:68]

        face_center_x = int(np.mean([face_contour[-1][0], face_contour[0][0], face_contour[-1][0], face_contour[1][0]]))
        face_width = int(np.mean([face_contour[-1][0] - face_contour[0][0], face_contour[-2][0] - face_contour[1][0]]))
        eyebrow_top = int(min(min(left_eyebrow[:, 1]), min(right_eyebrow[:, 1])))

        # print(face_width, eyebrow_top)

        hat_size_ref = face_width
        hat_x_ref = face_center_x
        hat_y_ref = eyebrow_top

        # cv2.line(img, (0, hat_y_ref), (img.shape[1], hat_y_ref), color=(0, 0, 255), thickness=1)
        # cv2.line(img, (hat_x_ref, 0), (hat_x_ref, img.shape[0]), color=(0, 0, 255), thickness=1)

        face_hat_factor = 1.6
        resized_hat_h = int(round(hat_rgb.shape[0] * hat_size_ref / hat_rgb.shape[1] * face_hat_factor))
        resized_hat_w = int(round(hat_rgb.shape[1] * hat_size_ref / hat_rgb.shape[1] * face_hat_factor))

        if resized_hat_h > target_y:
            resized_hat_h = target_y - 1
            continue

        hat_resized = cv2.resize(hat_rgb, (resized_hat_w, resized_hat_h))
        hat_mask = cv2.resize(alpha, (resized_hat_w, resized_hat_h))
        hat_mask_inv = cv2.bitwise_not(hat_mask)
        # cv2.imshow("hat_mask", hat_mask)
        # cv2.imshow("hat_mask_inv", hat_mask_inv)
        # cv2.waitKey(0)

        offset_ratio_h = -0.05
        offset_ratio_w = 0.19
        offset_h = int(target_h * offset_ratio_h)
        offset_w = int(target_w * offset_ratio_w)

        # ROI in the original img
        roi_y1 = hat_y_ref + offset_h - resized_hat_h
        roi_y2 = hat_y_ref + offset_h
        roi_x1 = hat_x_ref + offset_w - resized_hat_w // 2
        roi_x2 = hat_x_ref + offset_w + resized_hat_w // 2
        bg_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        # cv2.imshow("roi", bg_roi)
        # cv2.waitKey(0)

        # cv2.line(img, (0, roi_y1), (img.shape[1], roi_y1), color=(0, 255, 0), thickness=1)
        # cv2.line(img, (0, roi_y2), (img.shape[1], roi_y2), color=(0, 255, 0), thickness=1)

        hat_mask_inv = cv2.merge((hat_mask_inv, hat_mask_inv, hat_mask_inv))
        hat_mask_inv_copy = hat_mask_inv.copy()
        bg_roi = bg_roi.astype(float)
        hat_mask_roi = hat_mask_inv.astype(float) / 255
        # cv2.imshow("hat_mask_inv", hat_mask_inv)
        # cv2.waitKey(0)

        # put hat mask on ROI
        hat_mask_roi = cv2.resize(hat_mask_roi, (bg_roi.shape[1], bg_roi.shape[0]))  # in case of size difference due to rounding
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

if __name__ == "__main__":
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