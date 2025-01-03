import argparse

from classifier import HandClassifier
from data_manager import DataManager

parser = argparse.ArgumentParser(
    prog="Gesture Recogniser", description="Application for detecting hand gestures"
)

parser.add_argument(
    "--mode",
    type=str,
    choices=["collect-data", "train", "detect"],
    help="mode to run the detector in: [collect-data, train, detect]",
    default="detect",
)
args = parser.parse_args()

mode = args.mode


def run_detector():
    data_manager = DataManager()
    classifier = HandClassifier()

    if mode == "collect-data":
        print(
            "Enter the label for the object you will be collecting data for: ", end=" "
        )
        label = input()

        data_manager.collect_data(label)
    elif mode == "train":
        result_file = data_manager.process_data()
        print(f"Proccessed data saved as file: {result_file}")
        # classify
        classifier.train(result_file)
    elif mode == "detect":
        classifier.classify()


if __name__ == "__main__":
    run_detector()
