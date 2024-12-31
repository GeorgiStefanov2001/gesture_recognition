import glob
import os
import pickle
import uuid
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# default path where the collected images get saved
COLLECTED_IMGS_PATH = os.path.join(Path.cwd(), "data", "collected-images")
if not os.path.exists(COLLECTED_IMGS_PATH):
    os.makedirs(COLLECTED_IMGS_PATH)

# default path where the proccessed data gets saved
DATA_DIR = os.path.join(Path.cwd(), "data", "proccessed-data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


class DataManager:
    def __init__(self, collected_images_path=COLLECTED_IMGS_PATH, data_dir=DATA_DIR):
        self.collected_images_path = collected_images_path
        self.data_dir = data_dir

    def collect_data(self, label, dataset_size=100, camera_device_id=0):
        """
        Collect data from the device's camera in the form of images with a given dataset size.
        """
        label_dir = os.path.join(self.collected_images_path, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        cap = cv2.VideoCapture(camera_device_id)
        print(f"Collecting data for {label}...")

        while True:
            _, frame = cap.read()
            cv2.putText(
                frame,
                'Press "e" when you are ready to capture',
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) == ord("e"):
                break

        for i in range(dataset_size):
            _, frame = cap.read()
            cv2.imshow("frame", frame)

            cv2.waitKey(25)
            image_name = os.path.join(label_dir, f"{label}.{str(uuid.uuid1())}.jpg")
            cv2.imwrite(image_name, frame)

        cap.release()
        cv2.destroyAllWindows()

        return label_dir

    def process_data(self, debug=False, data_dir=None):
        """
        Process the data, transforming images into a list of values (points of interest of the hand), representing
        the fingers and their joints.
        """
        data = []
        labels = []

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        print("Proccessing data...")
        top_dirs = (
            [data_dir]
            if data_dir != None
            else [dir[0] for dir in os.walk(self.collected_images_path)]
        )  # 3-tuple, where first entry is dir name
        for dir_name in top_dirs:
            images = glob.glob(
                f"{os.path.join(self.collected_images_path, dir_name)}/*.jpg"
            )
            images = images[:1] if debug else images
            for img_path in images:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB
                )  # mediapipe requires rgb for landmark detection

                proccessed_img = hands.process(img_rgb)

                if proccessed_img.multi_hand_landmarks:
                    # create a list of landmarks
                    curr_data = []
                    for landmarks in proccessed_img.multi_hand_landmarks:
                        # display landmarks for debugging purposes
                        if debug:
                            mp_drawing.draw_landmarks(
                                img_rgb,
                                landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style(),
                            )

                        for idx in range(len(landmarks.landmark)):
                            # each landmark in the hand landmarks has 3 values - x, y, z
                            landmark_info = landmarks.landmark[idx]
                            x_coord = landmark_info.x
                            y_coord = landmark_info.y
                            curr_data.append(x_coord)
                            curr_data.append(y_coord)

                    # add the landmarks information for the current image to the aggregate list
                    data.append(curr_data)  # create list of lists
                    labels.append(dir_name.split("/")[-1])

                    if debug:
                        plt.figure()
                        plt.imshow(img_rgb)
        if debug:
            plt.show()

        # save the data list in a file
        proccessed_file_path = os.path.join(
            self.data_dir, f"data.{str(uuid.uuid1())}.pickle"
        )
        with open(proccessed_file_path, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)

        return proccessed_file_path
