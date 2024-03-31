import sys
import copy
import time
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import tensorflow as tf



def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    input_image = cv.resize(image, dsize=(input_size, input_size))  
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB
    input_image = input_image.reshape(-1, input_size, input_size, 3) 
    input_image = tf.cast(input_image, dtype=tf.uint8)  # uint8 / float32

    with tf.device('/cpu:0'):
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores

def draw_debug(image, elapsed_time, keypoint_score_th, keypoints, scores):
    debug_image = copy.deepcopy(image)

    index01, index02 = 0, 1
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 0, 2
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 1, 3
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 2, 4
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 0, 5
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 0, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 5, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 5, 7
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 7, 9
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 6, 8
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 8, 10
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 11, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 5, 11
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 11, 13
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 13, 15
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 6, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 12, 14
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    index01, index02 = 14, 16
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)

    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "fps",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "fps",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image


def main():
    cap_device = 0
    keypoint_score_th = 0.2

    cap_device = 'Media/dance.mp4'
    cap = cv.VideoCapture(cap_device)

    model_path = '/Users/valuepawn3244/MoveNet-Python-Example/Models/singlepose-lightning-tflite-float16.tflite'
    input_size = 192

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.2,
                       min_tracking_confidence=0.2)

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(frame)

        image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False  # To improve performance, optionally mark the image as not writeable.
        results = hands.process(image)

        keypoints, scores = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = 1000*(time.time() - start_time)
        elapsed_time = 1000/elapsed_time

        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints,
            scores,
        )

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # Calculate the x and y coordinates
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    # Draw a circle for each landmark
                    cv.circle(debug_image, (x, y), 4, (255, 255, 0), -1)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('MoveNet(singlepose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()