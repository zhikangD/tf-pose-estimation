import argparse
import logging
import time
import ast
import cv2
import common

import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
from balloon_mask import mask

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    t = time.time()
    humans = e.inference(image, scales=scales)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image,bodys = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # cv2.imshow('tf-pose-estimation result', image)
    # cv2.waitKey()
    maskimg = cv2.imread(args.image, -1)
    img2 = cv2.imread('./src/balloons.png',-1)
    for centers in bodys:
        if 0 in centers.keys():
            if 3 in centers.keys() and 4 in centers.keys():
                right_elbow = centers[3]
                right_wrist = centers[4]
                right_hand = (right_wrist[0]+int((right_wrist[0]-right_elbow[0])*0.2),
                              right_wrist[1]+int((right_wrist[1]-right_elbow[1])*0.2))
                if right_hand[1]<centers[0][1]:
                    maskimg = mask(maskimg, img2,right_elbow,right_wrist)
            if 6 in centers.keys() and 7 in centers.keys():
                left_elbow = centers[6]
                left_wrist = centers[7]
                left_hand = (left_wrist[0] + int((left_wrist[0] - left_elbow[0]) * 0.2),
                              left_wrist[1] + int((left_wrist[1] - left_elbow[1]) * 0.2))
                if left_hand[1]<centers[0][1]:
                    maskimg = mask(maskimg, img2, left_elbow, left_wrist)

    cv2.imshow('res', maskimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()