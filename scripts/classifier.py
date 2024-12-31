import pickle

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Classifier:
    def train(self, data_file_path):
        """
        Train the classifier with the data it posseses.
        """
        data_dict = {}
        with open(data_file_path, "rb") as f:
            data_dict = pickle.load(f)

        print("Training model with latest data...")

        # pad data coming in
        # this is needed in cases where mediapipe doesn't detect all 84 landmarks in both hands
        data = data_dict["data"]
        max_len = max(len(sample) for sample in data)
        padded_data = [sample + [0] * (max_len - len(sample)) for sample in data]

        # the classifier accepts numpy arrays (instead of default python lists)
        data = np.asarray(padded_data)
        labels = np.asarray(data_dict["labels"])

        # split into train set and test set in 80/20 fashion
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )

        # train Random Forest
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # get prediction
        label_predict = model.predict(x_test)

        # get accuracy score
        score = accuracy_score(label_predict, y_test)
        print(f"Accuracy score after training: {score * 100}")

        # save model
        with open("model.p", "wb") as f:
            pickle.dump({"model": model}, f)

    def classify(self, model_path="./model.p", camera_device_id=0):
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)

        model = model_dict["model"]

        cap = cv2.VideoCapture(camera_device_id)

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        while True:
            ret, frame = cap.read()
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                x_coords = []
                y_coords = []

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                for landmarks in results.multi_hand_landmarks:
                    for idx in range(len(landmarks.landmark)):
                        # each landmark in the hand landmarks has 3 values - x, y, z
                        landmark_info = landmarks.landmark[idx]
                        x_coord = landmark_info.x
                        y_coord = landmark_info.y

                        x_coords.append(x_coord)
                        y_coords.append(y_coord)

                # create data for prediction with x and y coords
                curr_data = [item for pair in zip(x_coords, y_coords) for item in pair]

                # pad data to 84 since that is the max the model has been trained with
                needed_padding_length = 84 - len(curr_data)
                padded_data = curr_data + [0] * needed_padding_length
                prediction = model.predict([np.asarray(padded_data)])
                predicted_label = prediction[0]

                # draw rectangle around detected shape
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10

                x2 = int(max(x_coords) * W) - 10
                y2 = int(max(y_coords) * H) - 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    predicted_label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

            cv2.imshow("Detecting gestures", frame)
            cv2.waitKey(1)
