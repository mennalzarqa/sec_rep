import cv2
import numpy as np
from matplotlib import pyplot as plt





before = cv2.imread('after.png')
after = cv2.imread('after2.png')

# before = cv2.bilateralFilter(before,9,75,75)
# after = cv2.bilateralFilter(after,9,75,75)

before = cv2.GaussianBlur(before,(5,5),0)
after = cv2.GaussianBlur(after,(5,5),0)


hsv_frame1 = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
hsv_frame2 = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)

# Pink color
low_pink = np.array([125, 100, 30])
high_pink = np.array([179, 255, 255])


# define range of white color in HSV
# change it according to your need !
lower_white = np.array([0,0,168], dtype=np.uint8)
upper_white = np.array([110,111,255], dtype=np.uint8)

# Threshold the HSV image to get only white colors
# Bitwise-AND mask and original image



pink_mask1 = cv2.inRange(hsv_frame1, low_pink, high_pink)
pink_mask2= cv2.inRange(hsv_frame2, low_pink, high_pink)

white_mask1 = cv2.inRange(hsv_frame1, lower_white, upper_white)
white_mask2 = cv2.inRange(hsv_frame2, lower_white, upper_white)

res1= cv2.bitwise_and(before,before, mask= white_mask1)
res2 = cv2.bitwise_and(after,after, mask= white_mask2)


res3= cv2.bitwise_and(before, before, mask=pink_mask1)
res4= cv2.bitwise_and(after, after, mask=pink_mask2)



cv2.imshow("before", res1)
cv2.imshow("after", res2)
cv2.imshow("before1", res3)
cv2.imshow("after2", res4)




cv2.imwrite('pink_a.jpg',res1)
cv2.imwrite('pink_b.jpg',res2)
# cv2.imwrite('white_a.jpg',white_a)
# cv2.imwrite('white_b.jpg',white_b)

cv2.waitKey(0)


# ###
# IN_MATCH_COUNT = 10
# img1 = cv2.imread(pink_a)          # queryImage
# img2 = cv2.imread(pink_b) # trainImage
# # Initiate SIFT detector
# sift = cv2.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# if len(good)>IN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w,d = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'gray')
# plt.show()



# cv2.waitKey(0)
