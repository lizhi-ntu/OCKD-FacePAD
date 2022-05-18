import os
import numpy as np
import cv2
import dlib


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
 
    def align(self, image, pos):
        leftEyeCenter = pos[0]
        rightEyeCenter = pos[1]
 
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
 
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
 
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                      int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
 
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)
 
        # return the aligned face
        return output

def shape_to_np(shape, dtype="int"): 
    # facial landmark coordinates: shape -> numpy array
    coords = np.zeros((5, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def video2face(in_dir, out_dir, detector, predictor, aligner):
    # video -> face image

    frm = 0  # a counter of frames with detected faces
    cap = cv2.VideoCapture(in_dir)
    while True:
        res, img = cap.read()
        if not res:
            break           
        frm += 1
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img_gray.shape
        rects = detector(img_gray, 0)
        if len(rects)>0:
            rect = rects[0]
            shape = predictor(img_gray, rect)
            shape = shape_to_np(shape)
            pos = np.array([0.5 * (shape[1] + shape[0]), 0.5 * (shape[3] + shape[2])]).astype(int)
            aligned_face = aligner.align(image=img, pos=pos)
            cv2.imwrite('{}/{}.jpg'.format(out_dir, int(frm)), aligned_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print('{}/{}.jpg'.format(out_dir, int(frm)))
    cap.release()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
aligner = FaceAligner()

# Please customize the script below >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
video_folder = 'a directory you store videos'
face_folder = 'a directory you store face images'
videos = os.listdir(video_folder)
for video in videos:
    video_name, _ = os.path.splitext(video)
    print(video_name)
    if not os.path.exists('{}/{}'.format(face_folder, video_name)):
        os.makedirs('{}/{}'.format(face_folder, video_name))
        in_dir = '{}/{}'.format(video_folder, video)
        out_dir = '{}/{}'.format(face_folder, video_name)
        video2face(in_dir=in_dir, out_dir=out_dir, detector=detector, predictor=predictor, aligner=aligner)
# Please customize the script above <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
