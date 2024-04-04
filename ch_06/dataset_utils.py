from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import glob
import numpy as np

import cv2
from PIL import Image
from imutils import video


DOWNSAMPLE_RATIO = 4
DATASET_PATH = "./images/"

class ImageDataset(Dataset):
    def __init__(self, dataset_path, image_transformations=None):
        self.transform = transforms.Compose(image_transformations)
        self.orig_files = sorted(glob.glob(os.path.join(dataset_path,'original') + "/*.*"))
        self.landmark_files = sorted(glob.glob(os.path.join(dataset_path,'landmarks') + "/*.*"))

    def __getitem__(self, index):

        orig_img = Image.open(self.orig_files[index % len(self.orig_files)])
        landmark_img = Image.open(self.landmark_files[index % len(self.landmark_files)])

        # flip images randomly
        if np.random.random() < 0.5:
            orig_img = Image.fromarray(np.array(orig_img)[:, ::-1, :], "RGB")
            landmark_img = Image.fromarray(np.array(landmark_img)[:, ::-1, :], "RGB")

        orig_img = self.transform(orig_img)
        landmark_img = self.transform(landmark_img)

        return {"A": orig_img, "B": landmark_img}

    def __len__(self):
        return len(self.orig_files)

def reshape_array(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def resize(image,img_width,img_height):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (img_width, img_height), interpolation = cv2.INTER_LINEAR)
        return image_resize

def rescale_frame(frame):
    dim = (256, 256)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def get_landmarks(black_image,gray,faces,predictor):
    for face in faces:
        detected_landmarks = predictor(gray, face).parts()
        landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

        jaw = reshape_array(landmarks[0:17])
        left_eyebrow = reshape_array(landmarks[22:27])
        right_eyebrow = reshape_array(landmarks[17:22])
        nose_bridge = reshape_array(landmarks[27:31])
        lower_nose = reshape_array(landmarks[30:35])
        left_eye = reshape_array(landmarks[42:48])
        right_eye = reshape_array(landmarks[36:42])
        outer_lip = reshape_array(landmarks[48:60])
        inner_lip = reshape_array(landmarks[60:68])

        color = (255, 255, 255)
        thickness = 3

        cv2.polylines(black_image, [jaw], False, color, thickness)
        cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
        cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
        cv2.polylines(black_image, [nose_bridge], False, color, thickness)
        cv2.polylines(black_image, [lower_nose], True, color, thickness)
        cv2.polylines(black_image, [left_eye], True, color, thickness)
        cv2.polylines(black_image, [right_eye], True, color, thickness)
        cv2.polylines(black_image, [outer_lip], True, color, thickness)
        cv2.polylines(black_image, [inner_lip], True, color, thickness)
    return black_image

def prepare_data(video_file_path, detector, predictor, num_samples=400, downsample_ratio = DOWNSAMPLE_RATIO):
    """
    Utility to prepare data for pix2pix based deepfake.
    Output is a set of directories with original frames
    and their corresponding facial landmarks
    Parameters:
        video_file_path : path to video to be analysed
        num_samples : number of frames/samples to be extracted
    """

    # create output directories
    os.makedirs(f'{DATASET_PATH}',exist_ok=True)
    os.makedirs(f'{DATASET_PATH}/original', exist_ok=True)
    os.makedirs(f'{DATASET_PATH}/landmarks', exist_ok=True)

    # get video capture object
    cap = cv2.VideoCapture(video_file_path)
    fps = video.FPS().start()

    # iterate through video frame by fame
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # resize frame
        frame_resize = cv2.resize(frame,
                                  None,
                                  fx=1 / downsample_ratio,
                                  fy=1 / downsample_ratio)

        # gray scale
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)

        # detect face
        faces = detector(gray, 1)

        # black background
        black_image = np.zeros(frame.shape, np.uint8)

        # Proceed only if face is detected
        if len(faces) == 1:
            black_image = get_landmarks(black_image,gray,faces,predictor)

            # Display the resulting frame
            count += 1
            cv2.imwrite(f"{DATASET_PATH}/original/{count}.png", frame)
            cv2.imwrite(f"{DATASET_PATH}/landmarks/{count}.png", black_image)
            fps.update()

            # stop after num_samples
            if count == num_samples:
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No face detected")

    fps.stop()
    print('Total time: {:.2f}'.format(fps.elapsed()))
    print('Approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()