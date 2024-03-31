#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument("--model_select", type=int, default=3)
    parser.add_argument("--keypoint_score", type=float, default=0.1)

    args = parser.parse_args()

    return args


def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    input_image = cv.resize(image, dsize=(input_size, input_size))  
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = tf.cast(input_image, dtype=tf.float32)  # uint8 / float32

    with tf.device('/cpu:0'):
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        keypoints_with_scores = np.squeeze(keypoints_with_scores)
        #hand_landmarks = interpreter.get_tensor(output_details[0]['index'])
    
    keypoints = []#np.zeros(17)
    scores = []
    '''for i in range(17):
        keypoint_x = int(hand_landmarks[0, i])
        keypoint_y = int(hand_landmarks[0, i+1])
        keypoints.append([keypoint_x, keypoint_y])
        scores.append(1)'''
        
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)
    
    return keypoints, scores


def main():
    ##################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    # カメラ準備 ###############################################################
    cap_device = 0#'Media/IMG_2415.MOV'
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    if model_select == 0:
        model_path = 'tflite/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
        input_size = 192
    elif model_select == 1:
        model_path = 'tflite/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
        input_size = 256
    elif model_select == 2:
        model_path = 'tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'
        input_size = 192
    elif model_select == 3:
        model_path = 'tflite/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'
        input_size = 256
    else:
        sys.exit(
            "*** model_select {} is invalid value. Please use 0-3. ***".format(
                model_select))
        
    model_path = 'Models/singlepose-lightning.tflite'
    input_size = 192

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    cv.namedWindow('MoveNet(singlepose) Demo')

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv.flip(frame, 1) 
        input_image = copy.deepcopy(frame)
        #frame = cv.resize(frame, dsize=(input_size, input_size))  

        keypoints = np.zeros(17)
        scores = np.zeros(17)
        keypoints, scores = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = 1000*(time.time() - start_time)
        elapsed_time = 1000/elapsed_time

        debug_image = draw_debug(
            input_image,
            elapsed_time,
            keypoint_score_th,
            keypoints,
            scores,
        )
        
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('MoveNet(singlepose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    # Line：鼻 → 左目
    index01, index02 = 0, 1
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右目
    index01, index02 = 0, 2
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左目 → 左耳
    index01, index02 = 1, 3
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右目 → 右耳
    index01, index02 = 2, 4
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 左肩
    index01, index02 = 0, 5
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右肩
    index01, index02 = 0, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 右肩
    index01, index02 = 5, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左肘
    index01, index02 = 5, 7
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肘 → 左手首
    index01, index02 = 7, 9
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右肘
    index01, index02 = 6, 8
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肘 → 右手首
    index01, index02 = 8, 10
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 右股関節
    index01, index02 = 11, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左股関節
    index01, index02 = 5, 11
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 左ひざ
    index01, index02 = 11, 13
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左ひざ → 左足首
    index01, index02 = 13, 15
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右股関節
    index01, index02 = 6, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右股関節 → 右ひざ
    index01, index02 = 12, 14
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右ひざ → 右足首
    index01, index02 = 14, 16
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)

    # Circle：各点
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    # 処理時間
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "fps",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "fps",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()