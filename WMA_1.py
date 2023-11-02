import cv2
import numpy as np

# img = cv2.imread('ball.png')

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower_red = np.array([0, 100, 100])
# upper_red = np.array([10, 255, 255])
# mask1 = cv2.inRange(hsv, lower_red, upper_red)
# lower_red = np.array([160, 100, 100])
# upper_red = np.array([179, 255, 255])
# mask2 = cv2.inRange(hsv, lower_red, upper_red)
# mask = cv2.bitwise_or(mask1, mask2)

# kernel = np.ones((5,5), np.uint8)
# noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# M = cv2.moments(noise)
# if M["m00"] != 0:
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])
# else:
#     cx, cy = 0, 0

# img_with_marker = cv2.circle(img.copy(), (cx, cy), 5, (0, 255, 0), -1)

# cv2.imshow('original', hsv)
# cv2.imshow('mask', mask)
# cv2.imshow('noise', noise)
# cv2.imshow('with marker', img_with_marker)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


########################################################


video = cv2.VideoCapture('bird.mp4')
video.open('bird.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

counter = 1

while True:
    success, frame_rgb = video.read()
    if not success:
        break
    print('klatka {} z {}'.format(counter, total_frames))
    counter = counter + 1

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([22, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5,5), np.uint8)
    noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    M = cv2.moments(noise)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    frame_with_marker = cv2.circle(frame_rgb.copy(), (cx, cy), 5, (0, 255, 0), -1)
    frame_with_marker = cv2.resize(frame_with_marker, (int(frame_width/2), int (frame_height/2)))

    cv2.imshow('result', frame_with_marker)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
result.release()
video.release()