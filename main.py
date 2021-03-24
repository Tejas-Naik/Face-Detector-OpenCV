import cv2
from random import randrange

#opening the xml file
trained_face_data = cv2.CascadeClassifier("hearcasecade_frontal_face_default.xml")

# image to deetct a face
# img = cv2.imread('TEJAS.JPG')
# img = cv2.imread('group.jpg')

# to capture video
webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read,frame = webcam.read()

    # converting image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # print(face_coordinates)

    # Draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)

    # Display the Image
    cv2.imshow("Python Face Detector", frame)
    key = cv2.waitKey(1)

    # stop if Q is pressed
    if key == 81 or key == 113:
        break

    # release the VideoCapture object
    


print("Code Completed")
