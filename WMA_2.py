import cv2
import numpy as np

coin_values = {"5 gr": 0.05, "5 zl": 5.0}

for i in range(1, 9):

    img = cv2.imread(f"Copy of Copy of tray{i}.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 250)

    #cv2.imshow("edge", edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 90,
                            minLineLength=50, maxLineGap=150)
    
    image_l = img
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imshow("obrazek", image_l)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    tray_area = cv2.contourArea(cnt)

    rect = cv2.minAreaRect(cnt)
    tray_width = rect[1][0]
    tray_height = rect[1][1]

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=100, param2=35, minRadius=10, maxRadius=50)

    num_5zl = 0
    num_5gr = 0
    total_value_on_tray = 0
    total_value_off_tray = 0
    coin_centers = []

    if circles is not None:
        circles = circles[0][:12]
        for circle in circles:
            x, y, r = circle
            area = np.pi * (r * r)
            if area >= 3320:
                num_5zl += 1
                label = "5 zl"
            else:
                num_5gr += 1
                label = "5 gr"
            center = (int(x), int(y))
            coin_centers.append(center)
            radius = int(r)
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.putText(img, label, (int(x-r), int(y-r-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            coin_area = 3320
            coin_radius = np.sqrt(coin_area / np.pi)
            coin_diameter = coin_radius * 2
            size_diff = (coin_diameter / tray_width) * 100

            if cv2.pointPolygonTest(cnt, center, False) <= 0:
                total_value_on_tray += coin_values[label]
            else:
                total_value_off_tray += coin_values[label]

    print(f"Tray {i}: {num_5zl} 5 zł coins, {num_5gr} 5 gr coins")
    print(f"Tray {i}: {total_value_on_tray:.2f} zł on the tray, {total_value_off_tray:.2f} zł off the tray")
    print(f"Tray {i}: 5 zł coins are {size_diff:.2f}% smaller than the tray\n")

    cv2.imshow(f"Tray{i}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
