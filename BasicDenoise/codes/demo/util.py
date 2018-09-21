import numpy as np
import cv2

def cv_window_show_np8uimg(np_img_u8,
                           win_w=300, win_h=-1,
                           win_x=50, win_y=30,
                           win_name='in',
                           wait_time=-1):
    # calculate the resize scale rate
    h, w, _ = np_img_u8.shape
    show_scale_w = win_w / w
    show_scale_h = show_scale_w
    if win_h > 0:
        show_scale_h = win_h / hw
    show_w = int(show_scale_w * w)
    show_h = int(show_scale_h * h)

    # resize to better visual
    show_img = cv2.resize(np.copy(np_img_u8), (show_w, show_h), interpolation=cv2.INTER_CUBIC)

    # hori arrange
    cv2.namedWindow(win_name, 1)
    cv2.moveWindow(win_name, win_x, win_y)
    cv2.imshow(win_name, show_img)
    if wait_time >= 0:
        cv2.waitKey(wait_time)

    return win_x, win_y, show_w, show_h


def show_np8uimg_for_in_gt_out(inI, gtI, outI,
                               win_w = 300, win_h = -1,
                               win_x =  50, win_y = 30,
                               win_dist = 30,
                               win_wait = -1,
                               win_layout='hori'):
    # hori arrange
    if win_layout == 'vert':
        x1, y1, w1, h1 = cv_window_show_np8uimg(inI,
                                                win_w = win_w,
                                                win_x = win_x,
                                                win_y = win_y,
                                                win_name = 'in')
        x2, y2, w2, h2 = cv_window_show_np8uimg(gtI,
                                                win_w = win_w,
                                                win_x = win_x,
                                                win_y = y1 + h1 + win_dist,
                                                win_name = 'gt')
        x3, y3, w3, h3 = cv_window_show_np8uimg(outI,
                                                win_w = win_w,
                                                win_x = win_x,
                                                win_y = y2 + h2 + win_dist,
                                                win_name = 'out',
                                                wait_time = win_wait)
    else:
        x1, y1, w1, h1 = cv_window_show_np8uimg(inI,
                                                win_w = win_w,
                                                win_x = win_x,
                                                win_y = win_y,
                                                win_name = 'in')
        x2, y2, w2, h2 = cv_window_show_np8uimg(gtI,
                                                win_w = win_w,
                                                win_x = x1 + w1 + win_dist,
                                                win_y = win_y,
                                                win_name = 'gt')
        x3, y3, w3, h3 = cv_window_show_np8uimg(outI,
                                                win_w = win_w,
                                                win_x = x2 + w2 + win_dist,
                                                win_y = win_y,
                                                win_name = 'out',
                                                wait_time = win_wait)
    return x3, y3, w3, h3