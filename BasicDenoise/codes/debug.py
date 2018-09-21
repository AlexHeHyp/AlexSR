import numpy as np
import cv2

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/media/sdc/heyp/data/video/bear/MP4-1.2M-H.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')

#bCreateWriter = False
#out = cv2.VideoWriter('/media/sdc/heyp/data/video/bear/MP4-6.0M-out.avi',
#                          fourcc, 20.0, (640, 480))

while(True):
    ret, frame = cap.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #  if bCreateWriter == False:
  #      out = cv2.VideoWriter('/media/sdc/heyp/data/video/bear/MP4-200k-H-out.avi',
  #                        fourcc, 20.0, (640, 480))
  #      bCreateWriter = True
  #  frame_flip = cv2.flip(frame, 0)
  #  out.write(frame_flip)

    win_name = 'frame'
    win_x, win_y = 50, 20
    cv2.namedWindow(win_name, 0)
    cv2.moveWindow(win_name, win_x, win_y)
    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()