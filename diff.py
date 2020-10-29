from skimage.measure import compare_ssim
import cv2
import numpy as np
from colour import Color
from PIL import Image,ImageColor
from PIL import Image , ImageFilter
# cap=cv2.VideoCapture(0)
# while 1:
#     ret,frame=cap.read()
#     cv2.imshow('Video',frame)
#     k = cv2.waitKey(3)
#     if k==ord('s') or k==ord('S'):
#         saved_frame=frame
#         break
#     elif k==27:
#         break
before = cv2.imread('after.png')
before = cv2.resize(before,(500, 500))
after=cv2.imread('after2.png')
# print(before.shape)
after = cv2.resize(after,(500, 500))
# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
# print(before_gray.shape)
# print(after_gray.shape)

# Compute SSIM between two images
(score, diff) = compare_ssim(after_gray, before_gray, full=True)
# print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

white =Color("white")
pink=Color("pink")
black=Color('black')
grey=Color('grey')

for (i,c) in enumerate(sorted_contours):
    area = cv2.contourArea(c)
    if area > 40:
        # x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        # cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        # cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        # # cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
        # for x  in range(len(contours)):
        if i==0:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)


            # # cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]

            # print('Average color (BGR): ',np.array(cv2.mean(img_crop)).astype(np.uint8))

            # img_crop_color=np.array(cv2.mean(img_crop)).astype(np.uint8)
            # if img_crop_color==white
            M = cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("Centroid = ", cx, ", ", cy)

            # center_px = np.array([4,3,2],dtype=np.uint8)
            # before.getpixel((cx, cy))
            # center=img_crop.getcolor(cy,cx)
            # PIL.Image.Image.getpixel(before,cy,cx)
            center_px=before[cy,cx]
            center_px1=after[cy,cx]
            print (center_px)
            print(center_px1)

            b1,g1,r1 = before[cy,cx]
            b2,g2,r2 = after[cy,cx]
            #before
            b_b = int(b1)
            g_b = int(g1)
            r_b = int(r1)
            #after
            b_a = int(b2)
            g_a = int(g2)
            r_a = int(r2)

            # r_b,g_b,b_b=cv.split(before)
            # r_a,g_a,b_a=cv.split(pixel)
            # r_a = img[a[1],a[0],2]
            # g_a = img[a[1],a[0],1]
            # b_a = img[a[1],a[0],0]

            if (r_b>220 and g_b>220 and b_b>220): #WHITE before
                if r_a>=199 and (20<=g_a<=192) and (133<=b_a<=203): # pink after
                    print('cured ')
                else :
                    print('damaged ')
            elif (r_b>=199) and (20<=g_b<=192) and (133<=b_b<=203) :#PINK before

                if (r_a>220 and g_a>220 and b_a>220): #WHITE after
                    print('bleched')
                else :
                    print('damaged ')
            elif  r_b>=199 and (20<=g_b<=192) and (133<=b_b<=203): #PINK before
                if r_a>=199 and (20<=g_a<=192) and (133<=b_a<=203): # pink after
                    print('growth')
                else:
                    print('damaged ')

            elif  r_b>220 and g_b>220 and b_b>220: #WHITE before
                if  r_a>220 and g_a>220 and b_a>220: #WHITE after
                    print('growth ')
                else :
                    print('damaged')
            else :
                print('growth')

            #anything else

            # res=(img_crop-img_crop2)
            # res=np.mean(res)
            # print(res)
            # if res>0:
            #     print('positive')
            # elif res<0 :
            #     print('negative')
            # elif res==0:
            #     print('zero')
            # if (img_crop2-img_crop1):
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)

            cv2.imshow("img_crop_before0",img_crop)
            cv2.imshow('img_crop_after0',img_crop2)
            cv2.imwrite('contour_0_b.png',img_crop)
            cv2.imwrite('contour_0_a.png',img_crop2)


        elif i==1:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)

            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]

            # print('Average color (BGR): ',np.array(cv2.mean(img_crop)).astype(np.uint8))

            # img_crop_color=np.array(cv2.mean(img_crop)).astype(np.uint8)
            # if img_crop_color==white
            M = cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("Centroid = ", cx, ", ", cy)

            # center_px = np.array([4,3,2],dtype=np.uint8)
            # before.getpixel((cx, cy))
            # center=img_crop.getcolor(cy,cx)
            # PIL.Image.Image.getpixel(before,cy,cx)
            center_px=before[cy,cx]
            center_px1=after[cy,cx]
            print (center_px)
            print (center_px1)

            b1,g1,r1 = before[cy,cx]
            b2,g2,r2 = after[cy,cx]
            #before
            b_b = int(b1)
            g_b = int(g1)
            r_b = int(r1)
            #after
            b_a = int(b2)
            g_a = int(g2)
            r_a = int(r2)

            # r_b,g_b,b_b=cv.split(before)
            # r_a,g_a,b_a=cv.split(pixel)
            # r_a = img[a[1],a[0],2]
            # g_a = img[a[1],a[0],1]
            # b_a = img[a[1],a[0],0]

            if (r_b>220 and g_b>220 and b_b>220): #WHITE before
                if r_a>=199 and (20<=g_a<=192) and (133<=b_a<=203): # pink after
                    print('cured ')
                else :
                    print('damaged ')
            elif (r_b>=199) and (20<=g_b<=192) and (133<=b_b<=203) :#PINK before
                if (r_a>220 and g_a>220 and b_a>220): #WHITE after
                    print('bleched')
                else :
                    print('damaged ')
            elif  r_b>=199 and (20<=g_b<=192) and (133<=b_b<=203): #PINK before
                if r_a>=199 and (20<=g_a<=192) and (133<=b_a<=203): # pink after
                    print('growth')
                else:
                    print('damaged ')

            elif  r_b>220 and g_b>220 and b_b>220: #WHITE before
                if  r_a>220 and g_a>220 and b_a>220: #WHITE after
                    print('growth ')
                else :
                    print('damaged')
            else :
                print('growth')
            #anything else

            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]

            cv2.imshow("img_crop_before1",img_crop)
            cv2.imshow('img_crop_after1',img_crop2)
            cv2.imwrite('contour_one_b.png',img_crop)
            cv2.imwrite('contour_one_a.png',img_crop2)
        elif i==2:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            img_crop=before[y:y+h, x:x+w]
            img_crop2=after[y:y+h, x:x+w]


            # img_crop_color=np.array(cv2.mean(img_crop)).astype(np.uint8)
            # if img_crop_color==white
            M = cv2.moments(contours[i])
            # if M["m00"] == 0:
            #     continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("Centroid = ", cx, ", ", cy)

            center_px = np.array([4,3,2],dtype=np.uint8)
            # before.getpixel((cx, cy))
            # center=img_crop.getcolor(cy,cx)
            # PIL.Image.Image.getpixel(before,cy,cx)
            center_px=before[cy,cx]
            center_px1=after[cy,cx]
            print (center_px)
            print(center_px1)

            b1,g1,r1 = before[cy,cx]
            b2,g2,r2 = after[cy,cx]
            #before
            b_b = int(b1)
            g_b = int(g1)
            r_b = int(r1)
            #after
            b_a = int(b2)
            g_a = int(g2)
            r_a = int(r2)

            # r_b,g_b,b_b=cv.split(before)
            # r_a,g_a,b_a=cv.split(pixel)
            # r_a = img[a[1],a[0],2]
            # g_a = img[a[1],a[0],1]
            # b_a = img[a[1],a[0],0]

            if (r_b>220 and g_b>220 and b_b>220): #WHITE before
                if (r_a>=86) and (g_a>=0) and (b_a>=25): # pink after
                    print('cured ')
                else :
                    print('damaged ')
            elif r_a>=86 and (g_a>=0) and (b_a>=25) :#PINK before
                if (r_a>220 and g_a>220 and b_a>220): #WHITE after
                    print('bleched')
                else :
                    print('damaged ')
            elif  r_a>=86 and (g_a>=0) and (b_a>=25): #PINK before
                if r_a>=86 and (g_a>=0) and (b_a>=25): # pink after
                    print('growth')
                else:
                    print('damaged ')

            elif  r_b>220 and g_b>220 and b_b>220: #WHITE before
                if  r_a>220 and g_a>220 and b_a>220: #WHITE after
                    print('growth ')
                else :
                    print('damaged')
            else :
                print('growth')

            #anything else
            cv2.imshow("img_crop_before2",img_crop)
            cv2.imshow('img_crop_after2',img_crop2)
            cv2.imwrite('contour_two_b.png',img_crop)
            cv2.imwrite('contour_two_a.png',img_crop2)

        # img_sub=cv2.subtract(img_crop,img_crop2)
        # print(img_sub)

        # cv2.imshow("img_crop_before",img_crop[c])
        # cv2.imshow('img_crop_after',img_crop2[c])
        # cv2.waitKey(0)
        # b,g,r = cv2.split(img_crop)
        # print("b=",b)
        # print("g=",g)
        # print("r=",r)


cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)
