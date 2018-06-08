###################### LIBRARIES ######################

import cv2
import numpy as np
import os, os.path, time
import pandas as pd

##################### PARAMETERS #####################

# Starting time
t0 = time.time()
# Path of dataset
datasetPATH = '/media/gryphonlab/8847-9F6F/Ioannis/RML_Dataset/'
# Path to save extracted spectrograms
targetPATH = '/home/gryphonlab/Ioannis/Works/RML/InputFaces/'
# Path to spectrograms
spectrogramsPATH = '/home/gryphonlab/Ioannis/Works/RML/InputSpectrograms/'
# Sample rate of audio
audio_rate = 16000
# Sample rate of video
video_rate = 30
# Number of video frames for one spectrogram
num_frames = 11
# Stride between consecutive video frames
stride_frame = 2
# Size of cropped face-image
width, height = 110, 150
# Threshold for keeping a center movement per frame
c_threshold = 200
# Threshold for keeping a center vs other
d_threshold = 100000
# Video frame threshold
max_frame = 99000
# Threshold of maximum distance between eyes center and face center
face_eye_threshold = 20
# Minimum distance of eyes
eyes_threshold = 20
# Distance threshold of eyes for allowing scaling
distance_threshold = 40

###################### FUNCTIONS #####################


### Function for creating folders ###
def create_folder(PATH):
    
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print ('Error: Creating directory' + PATH)


### Function for calculating desired frames of a video according to audio ###
def calculate_desired_frames(times):

    desired_frames = list()

    for i in range(np.shape(times)[0]):

        for f in range(num_frames):
            
            desired_frame = int(round(times[i]*video_rate)) + stride_frame*(f-int((num_frames-1)/2))

            if not(desired_frame in desired_frames):
                desired_frames.append(desired_frame)

    desired_frames.append(max_frame)
    desired_frames.sort()
    #print(desired_frames)

    return desired_frames


### Function to crop image tracking the face ###
def crop_image(image, precenter, scale):

    # Set face tracking type
    cascPATH_face = '/home/gryphonlab/Ioannis/Works/MODELS/Haar-Cascade/haarcascade_frontalface_default.xml'
    cascPATH_eye = '/home/gryphonlab/Ioannis/Works/MODELS/Haar-Cascade/haarcascade_eye.xml'
    face_cascade = cv2.CascadeClassifier(cascPATH_face) 
    eye_cascade = cv2.CascadeClassifier(cascPATH_eye)
    # Copy image to show online cropping window
    img = image.copy()
    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces (-eyes) in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5)
    # Track speaker
    center, index = select_window_face(faces, precenter)
    # Calculate crop rectangle
    print(center)
    a = center[0] - int(width/2)
    b = center[1] - int(height/2)

    # Draw proposed center, face rectangle and crop rectangle
    cv2.circle(img,tuple(center),2,(0,255,0),2)
    if index >= 0:
        x, y, w, h = faces[index]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img,(a,b),(a+width,b+height),(0,0,255),2)
    
    #roi_gray = gray[b:b+int(round(height/2)), a:a+width]
    #roi_color = img[b:b+int(round(height/2)), a:a+width]
    roi_gray = gray[y:y+int(round(h/2)), x:x+w]
    roi_color = img[y:y+int(round(h/2)), x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.01, minNeighbors=5)
    scale, index1, index2 = select_window_eyes(eyes, scale)

    if index1 != -1:
    
        (ex1,ey1,ew1,eh1) = eyes[index1]
        cv2.rectangle(roi_color,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),2)
        (ex2,ey2,ew2,eh2) = eyes[index2]
        cv2.rectangle(roi_color,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
    
    # Show results
    cv2.imshow('Image',img)
    cv2.waitKey(1)
    # Crop initial image to selected face
    a = int(round(scale*center[0])) - int(width/2)
    b = int(round(scale*center[1])) - int(height/2)
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    face = image[b:b+height, a:a+width]

    return face, center, scale


### Function to select the best window before cropping
def select_window_face(faces, precenter):

    # Case[1] of detecting no 'face'
    if np.shape(faces)[0] == 0:
        # Proposed center
        center = precenter
        # False value for index in faces of selected window
        index = -1
    
    else:

        # Index in faces of selected window
        index = 0
        # Case[2] of detecting many 'faces'
        if np.shape(faces)[0] > 1:
            
            # Starting default distance
            dmin = d_threshold
            i = 0

            # Decide which to keep
            for (x,y,w,h) in faces:

                # Compute center
                xc = int(round((2*x+w)/2))
                yc = int(round((2*y+h)/2))
                d = np.linalg.norm(np.array([xc, yc])-np.array(precenter))
                # Change appropriately min and index
                if d < dmin:
                    dmin = d
                    index = i

                i += 1

            # Take values for proposed center
            index = int(index)
            x, y, w, h = faces[index]

        # Case[3] of detecting exactly one 'face'
        else:

            # Take values for proposed center
            x, y, w, h = faces[0]
            xc = int(round((2*x+w)/2))
            yc = int(round((2*y+h)/2))
            dmin = np.linalg.norm(np.array([xc, yc])-np.array(precenter))
            
        # Proposed centre
        xc = int(round((2*x+w)/2))
        yc = int(round((2*y+h)/2))
        # Check distance with precenter threshold
        if dmin < c_threshold:
            # Proposed center is accepted
            center = [xc, yc]
        else:
            # Proposed center is discarded, keep precenter
            center = precenter

    return center, index


def select_window_eyes(eyes, scale):

    xcenter = int(round((width)/2))
    ycenter = int(round((height)/2))
    updateFlag = False

    if np.shape(eyes)[0] >= 2:

        xmin = ymin = face_eye_threshold
        for i in range(np.shape(eyes)[0]):

            (ex1,ey1,ew1,eh1) = eyes[i]
            center1 = [int(round((2*ex1+ew1)/2)),int(round((2*ey1+eh1)/2))]

            for j in range(np.shape(eyes)[0]-1,i,-1):

                (ex2,ey2,ew2,eh2) = eyes[j]
                center2 = [int(round((2*ex2+ew2)/2)),int(round((2*ey2+eh2)/2))]

                eyes_center = [int(round((center1[0]+center2[0])/2)), int(round((center1[1]+center2[1])/2))]

                if abs(xcenter-eyes_center[0]) < xmin and abs(ycenter-eyes_center[1]) < ymin and abs(center1[0]-center2[0]) > eyes_threshold:

                    xmin = abs(xcenter-eyes_center[0])
                    ymin = abs(ycenter-eyes_center[1])
                    index1 = i
                    index2 = j
                    updateFlag = True

    if updateFlag:

        (ex1,ey1,ew1,eh1) = eyes[index1]
        center1 = np.array([int(round((2*ex1+ew1)/2)),int(round((2*ey1+eh1)/2))])
        (ex2,ey2,ew2,eh2) = eyes[index2]
        center2 = np.array([int(round((2*ex2+ew2)/2)),int(round((2*ey2+eh2)/2))])
        distance = int(round(np.linalg.norm(center1-center2)))
        print(distance)
        if distance > distance_threshold:
            scale = 55/distance

    else:

        index1, index2 = -1, -1
        
    return scale, index1, index2


### Function for running videos to capture frames ###
def run_video(videoPATH, facesPATH, times):

    if os.path.isfile(videoPATH):

        # Folder for frames of current video
        create_folder(facesPATH)
        # Calculate desired video frames according to audio
        desired_frames = calculate_desired_frames(times)
        # Playing video from file
        cap = cv2.VideoCapture(videoPATH)
        # current frame0.jpg
        currentFrame = 0
        # Position in desired frames list
        currentPosition = 0
        # Capture frame-by-frame
        success, frame = cap.read()
        # Initial precenter doesn't exist (center of frame)
        precenter = [int(round(320/2)), int(round(240/2))]
        # Initial scale factor equal to 1
        scale = 1

        while success:

            if (currentFrame == desired_frames[currentPosition]):

                # Fix frame size
                frame = cv2.resize(frame, (320,240))
                # Function to crop face from frame
                face, precenter, scale = crop_image(frame, precenter, scale)
                # Save current cropped image in .jpg file
                name = str(facesPATH) + '/frame' + str(currentFrame) + '.jpg'
                cv2.imwrite(name, face)
                # Move one position along desired frames list
                currentPosition += 1
            
            # To stop duplicate images
            print('Current Frame: Frame'+str(currentFrame))
            currentFrame += 1
            # Try capture next frame
            success, frame = cap.read()

        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print('Done')

    pass


######################## MAIN ########################


def main():

    speakers = os.listdir(datasetPATH)

    for speaker in speakers:
        
        sessionPATH = datasetPATH + speaker + '/'
        languages = os.listdir(sessionPATH)
        
        if len(languages) < 7:
            
            for language in languages:
                
                if language != 'Thumbs.db':
                    
                    speakerPATH = sessionPATH + language + '/'
                    videos = os.listdir(speakerPATH)
                    
                    for vid in videos:
                        
                        if vid[-3:] == 'avi':
                        
                            videoPATH = speakerPATH + vid
                            spectrograms = os.listdir(spectrogramsPATH + speaker + '/' + language + '/' + vid[:-4] + '/')
                            times = list()

                            for spectrogram in spectrograms:

                                t = float(spectrogram[2:-4])
                                times.append(t)

                            times = np.array(times)
                            facesPATH = targetPATH + speaker + '/' + language + '/' + vid[:-4] + '/'
                            create_folder(facesPATH)
                            run_video(videoPATH, facesPATH, times)

        else:
                    
            videos = languages
            
            for vid in videos:
                        
                if vid[-3:] == 'avi':
                        
                    videoPATH = sessionPATH + vid
                    spectrograms = os.listdir(spectrogramsPATH + speaker + '/' +vid[:-4] + '/')
                    times = list()

                    for spectrogram in spectrograms:

                        t = float(spectrogram[2:-4])
                        times.append(t)

                    times = np.array(times)
                    times.sort()
                    facesPATH = targetPATH + speaker + '/' + vid[:-4] + '/'
                    create_folder(facesPATH)
                    run_video(videoPATH, facesPATH, times)

        
    # Execution time
    print( 'Execution time of extractFace.py [sec]: ' + str(time.time() - t0) )

# Control runtime
if __name__ == '__main__':
    main()