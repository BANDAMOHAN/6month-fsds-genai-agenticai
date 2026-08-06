from flask import Flask, request, send_file, render_template
import cv2
import mediapipe as mp
import numpy as np
import io

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    color=(0,255,0)
)

transformation_matrix = np.array([
    [1.5,0,0],
    [0,1.5,0],
    [0,0,1]
])

def transform_3d_face(image, landmarks):

    transformed_landmarks = np.matmul(
        landmarks,
        transformation_matrix.T
    )

    transformed_image = image.copy()

    for i in range(transformed_landmarks.shape[0]):

        x,y,_ = transformed_landmarks[i]

        x=int(x*image.shape[1])
        y=int(y*image.shape[0])

        cv2.circle(
            transformed_image,
            (x,y),
            1,
            (255,0,0),
            -1
        )

    return transformed_image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process",methods=["POST"])
def process():

    file=request.files["image"]

    image=np.frombuffer(file.read(),np.uint8)

    image=cv2.imdecode(image,cv2.IMREAD_COLOR)

    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    results=face_mesh.process(rgb)

    transformed_image=image.copy()

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks=np.zeros((468,3),dtype=np.float32)

            for i,lm in enumerate(face_landmarks.landmark):

                landmarks[i]=[
                    lm.x,
                    lm.y,
                    lm.z
                ]

            transformed_image=transform_3d_face(
                image,
                landmarks
            )

            mp_drawing.draw_landmarks(
                transformed_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    _,buffer=cv2.imencode(".jpg",transformed_image)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/jpeg"
    )

if __name__=="__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
