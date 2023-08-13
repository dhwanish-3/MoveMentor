from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import cv2
import numpy as np
import time
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

interpreter = tf.lite.Interpreter(
    model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()
array = np.empty((0,))

file_counter = 0
file_table = []
UPLOAD_FOLDER = os.path.join(app.root_path, 'vid')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

NP_FOLDER = os.path.join(app.root_path, 'np')
if not os.path.exists(NP_FOLDER):
    os.makedirs(NP_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
start_stream = False


EDGES = {

    (0, 2): 'm',
    (2, 4): 'm',
    (1, 3): 'c',
    (3, 5): 'c',
    (0, 1): 'y',
    (0, 6): 'm',
    (1, 7): 'c',
    (6, 7): 'y',
    (6, 8): 'm',
    (8, 10): 'm',
    (7, 9): 'c',
    (9, 11): 'c'
}


def draw_key_points(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold, green):
    y, x, c = frame.shape

    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    if green:
        hue = (0, 255, 0)
    else:
        hue = (0, 0, 255)
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)),
                     (int(x2), int(y2)), hue, 2)

def time_frame(s, e):

    cap = cv2.VideoCapture('vid/happy_dance.mp4')
    # cap = cv2.VideoCapture('output.mp4')
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_delay = 1 / frame_rate
    codec = cv2.VideoWriter_fourcc(*'H264')
    output_file = 'output.mp4'
    fps = 30
    frame_size = (640, 480)
    out = cv2.VideoWriter(output_file, codec, fps, frame_size)
    max_size = (s-e) * frame_count + 10
    buffer_array = np.empty((max_size, 1, 12, 3))
    j = 0

    camera = cv2.VideoCapture(0)
    for i in range(s, e, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        cam_success, cam_frame = camera.read()
        success, frame = cap.read()

        if not success:
            break

        out.write(cam_frame)
        img = tf.image.resize_with_pad(
            np.expand_dims(cam_frame, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        input_details = interpreter.get_input_details() 
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(
            input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(
            output_details[0]['index'])[:, :, 5:, :]
        buffer_array[j] = keypoints_with_scores
        j = j + 1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # trying to match the speed
        time.sleep(frame_delay/2)
    start_stream = False
    cap.release()
    camera.release()
    buffer_path = os.path.join(app.root_path, 'buffer.npy')
    # np_path = os.path.join(NP_FOLDER, str(file_counter)+'.npy')
    np.save(buffer_path, buffer_array)


def store_video():
    video_path = file_table[file_counter]
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    array = np.empty((frame_count, 1, 12, 3))
    i = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(
            input_details[0]['index'], np.array(input_image))
        interpreter.invoke()

        key_points_with_scores = interpreter.get_tensor(
            output_details[0]['index'])[:, :, 5:, :]
        array[i] = key_points_with_scores
        i = i + 1

    np_path = os.path.join(NP_FOLDER, str(file_counter)+'.npy')
    np.save(np_path, array)

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recording')
def recording():
    return render_template('recording.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video_file' not in request.files:
        return jsonify({"messsage": "No file present"}), 400

    file = request.files['video_file']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_table.append(file_path)

        store_video()
        return redirect(url_for('recording'))
        # return render_template('recording.html')
        # return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/video')
def video():
    return Response(time_frame(0, 400), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_video', methods=['POST'])
def save_video():
    video_file = request.files['video']
    RECORDED_FOLDER = os.path.join(app.root_path, 'recorded')
    if not os.path.exists(RECORDED_FOLDER):
        os.makedirs(RECORDED_FOLDER)
    if video_file:
        video_file.save('recorded/recorded_video.webm')
        return "Video saved successfully"
    return "Video saving failed"

if __name__ == '__main__':
    app.run(debug=True)