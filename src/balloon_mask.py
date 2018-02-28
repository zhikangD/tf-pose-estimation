import cv2
from scipy.spatial import distance

def mask(origimg,balloonimg,elbow, wrist):
    # Create the mask for the mustache
    img1 = origimg
    img2 = balloonimg
    orig_mask = img2[:, :, 3]
    
    # Create the inverted mask for the mustache
    orig_mask_inv = cv2.bitwise_not(orig_mask)
    imgballoon = img2[:, :, 0:3]
    origBalloonHeight, origBalloonWidth = imgballoon.shape[:2]
    # mustacheWidth = int(origMustacheWidth)
    # mustacheHeight = int(origMustacheHeight)
    balloonratio = origBalloonHeight / origBalloonWidth
    hand = (wrist[0] + int((wrist[0] - elbow[0]) * 0.4),
                  wrist[1] + int((wrist[1] - elbow[1]) * 0.4))
    arml = int(distance.euclidean(elbow, wrist))
    balloonwidth, balloonheight = 3 * arml, int(3 * arml * balloonratio)
    balloon = cv2.resize(imgballoon, (balloonwidth, balloonheight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (balloonwidth, balloonheight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (balloonwidth, balloonheight), interpolation=cv2.INTER_AREA)
    x1 = hand[0] - int(balloonwidth / 2)
    x2 = x1 + balloonwidth
    y1 = hand[1] - balloonheight
    y2 = hand[1]
    bx1, by1 = 0, 0
    bx2, by2 = balloonwidth, balloonheight
    w = img1.shape[1]
    h = img1.shape[0]
    if x1 < 0:
        bx1 = bx1 - x1
        x1 = 0
    if y1 < 0:
        by1 = by1 - y1
        y1 = 0
    if x2 > w:
        bx2 = bx2 - (x2 - w)
        x2 = w
    if y2 > h:
        by2 = by2 - (y2 - h)
        y2 = h
    # Re-calculate the width and height of the mustache image
    # mustacheWidth = x2 - x1
    # mustacheHeight = y2 - y1
    balloon = balloon[by1:by2, bx1:bx2]
    mask = mask[by1:by2, bx1:bx2]
    mask_inv = mask_inv[by1:by2, bx1:bx2]
    
    roi = img1[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(balloon, balloon, mask=mask)
    dst = cv2.add(roi_bg, roi_fg)
    img1[y1:y2, x1:x2] = dst
    return  img1