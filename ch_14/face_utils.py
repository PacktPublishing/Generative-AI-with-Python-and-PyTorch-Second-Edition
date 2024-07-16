import cv2
import numpy as np
import face_recognition
from skimage import transform

from constants import landmarks_2D


def get_align_mat(face):
    mat = transform.SimilarityTransform()
    mat.estimate(np.array(face.landmarksAsXY()[17:]), landmarks_2D)
    return mat.params[0:2]


# Extraction class to identify and crop out the face from an image
class Extract(object):
    def extract(self, image, face, size):
        if face.landmarks is None:
            print("Warning! landmarks not found. Switching to crop!")
            return cv2.resize(face.image, (size, size))

        alignment = get_align_mat(face)
        return self.transform(image, alignment, size, padding=48)

    def transform(self, image, mat, size, padding=0):
        mat = mat * (size - 2 * padding)
        mat[:, 2] += padding
        return cv2.warpAffine(image, mat, (size, size))


# Filter class to check if a given image has a specific identity or not
class FaceFilter():
    def __init__(self, reference_file_path, threshold=0.65):
        """
        Works only for single face images
        """
        image = face_recognition.load_image_file(reference_file_path)
        self.encoding = face_recognition.face_encodings(image)[0]
        self.threshold = threshold

    def check(self, detected_face):
        encodings = face_recognition.face_encodings(np.array(detected_face.image))
        if len(encodings) > 0:
            encodings = encodings[0]
            score = face_recognition.face_distance([self.encoding], encodings)
        else:
            print("No faces found in the image!")
            score = 0.8
        return score <= self.threshold


class Convert():
    def __init__(self, encoder,
                 blur_size=2,
                 seamless_clone=False,
                 mask_type="facehullandrect",
                 erosion_kernel_size=None,
                 **kwargs):
        self.encoder = encoder

        self.erosion_kernel = None
        if erosion_kernel_size is not None:
            self.erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))

        self.blur_size = blur_size
        self.seamless_clone = seamless_clone
        self.mask_type = mask_type.lower()

    def patch_image(self, image, face_detected):
        size = 64
        image_size = image.shape[1], image.shape[0]

        mat = np.array(get_align_mat(face_detected)).reshape(2, 3) * size

        new_face = self.get_new_face(image, mat, size)

        image_mask = self.get_image_mask(
            image, new_face, face_detected, mat, image_size)

        return self.apply_new_face(image,
                                   new_face,
                                   image_mask,
                                   mat,
                                   image_size,
                                   size)

    def apply_new_face(self,
                       image,
                       new_face,
                       image_mask,
                       mat,
                       image_size,
                       size):
        base_image = np.copy(image)
        new_image = np.copy(image)

        cv2.warpAffine(new_face, mat, image_size, new_image,
                       cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT)

        outimage = None
        if self.seamless_clone:
            masky, maskx = cv2.transform(np.array([size / 2, size / 2]).reshape(
                1, 1, 2), cv2.invertAffineTransform(mat)).reshape(2).astype(int)
            outimage = cv2.seamlessClone(new_image.astype(np.uint8), base_image.astype(
                np.uint8), (image_mask * 255).astype(np.uint8), (masky, maskx), cv2.NORMAL_CLONE)
        else:
            foreground = cv2.multiply(image_mask, new_image.astype(float))
            background = cv2.multiply(
                1.0 - image_mask, base_image.astype(float))
            outimage = cv2.add(foreground, background)

        return outimage

    def get_new_face(self, image, mat, size):
        face = cv2.warpAffine(image, mat, (size, size))
        face = np.expand_dims(face, 0)
        new_face = self.encoder(face / 255.0)[0]

        return np.clip(new_face * 255, 0, 255).astype(image.dtype)

    def get_image_mask(self, image, new_face, face_detected, mat, image_size):

        face_mask = np.zeros(image.shape, dtype=float)
        if 'rect' in self.mask_type:
            face_src = np.ones(new_face.shape, dtype=float)
            cv2.warpAffine(face_src, mat, image_size, face_mask,
                           cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT)

        hull_mask = np.zeros(image.shape, dtype=float)
        if 'hull' in self.mask_type:
            hull = cv2.convexHull(np.array(face_detected.landmarksAsXY()).reshape(
                (-1, 2)).astype(int)).flatten().reshape((-1, 2))
            cv2.fillConvexPoly(hull_mask, hull, (1, 1, 1))

        if self.mask_type == 'rect':
            image_mask = face_mask
        elif self.mask_type == 'faceHull':
            image_mask = hull_mask
        else:
            image_mask = ((face_mask * hull_mask))

        if self.erosion_kernel is not None:
            image_mask = cv2.erode(
                image_mask, self.erosion_kernel, iterations=1)

        if self.blur_size != 0:
            image_mask = cv2.blur(image_mask, (self.blur_size, self.blur_size))

        return image_mask


# Entity class
class DetectedFace(object):
    def __init__(self, image, x, w, y, h, landmarks):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarks = landmarks

    def landmarksAsXY(self):
        return [(p.x, p.y) for p in self.landmarks.parts()]
