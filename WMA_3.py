import cv2

# Zadanie 2: Przygotuj zdjęcia elementu tak by można było z nich wyciągnąć jak najwięcej cech przy pomocy SIFT i wyodrębnij ich cechy.

sift = cv2.SIFT_create()

img1 = cv2.imread('photo_1.jpg')
img2 = cv2.imread('photo_2.jpg')
img3 = cv2.imread('photo_3.jpg')

img_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img_3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)
kp3, des3 = sift.detectAndCompute(img_3, None)

# print(f"Found {len(des1)} keypoints in image 1")
# print(f"Found {len(des2)} keypoints in image 2")
# print(f"Found {len(des3)} keypoints in image 3")

# img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(
#         0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(
#         0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img3_kp = cv2.drawKeypoints(img3, kp3, None, color=(
#         0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("Image 1 with keypoints", img1_kp)
# cv2.imshow("Image 2 with keypoints", img2_kp)
# cv2.imshow("Image 3 with keypoints", img3_kp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Zadanie 3: Dla każdej klatki filmu wybierz to zdjęcie które ma najwięcej dopasowań i wygeneruj nową klatkę na której jest wizualizacja dopasowań i oznaczenie szukanego elementu.

cap = cv2.VideoCapture('video_1.MOV')

# bf = cv2.BFMatcher()

# out = cv2.VideoWriter('video_1_matched.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret:
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         kp, des = sift.detectAndCompute(frame_gray, None)

#         matches = bf.match(des1, des)

#         best_match = min(matches, key=lambda x: x.distance)

#         img_matched = cv2.drawMatches(img1, kp1, frame, kp, [best_match], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#         img_matched = cv2.resize(img_matched, (1280, 720))

#         out.write(img_matched)

#         cv2.imshow('frame', img_matched)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()


# # Zadanie 4: ORB


orb = cv2.ORB_create()

kp_orb, des_orb = orb.detectAndCompute(img_1, None)

bf = cv2.BFMatcher()

out = cv2.VideoWriter('video_1_matched.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, des = orb.detectAndCompute(frame_gray, None)

        matches = bf.match(des_orb, des)

        best_match = min(matches, key=lambda x: x.distance)

        img_matched = cv2.drawMatches(img1, kp_orb, frame, kp, [best_match], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_matched = cv2.resize(img_matched, (1280, 720))

        out.write(img_matched)

        cv2.imshow('matched', img_matched)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

