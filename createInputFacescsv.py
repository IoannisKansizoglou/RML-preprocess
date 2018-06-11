
########################### LIBRARIES ##########################

import os, os.path, time
import pandas as pd
import numpy as np

########################## PARAMETERS ##########################


# Starting time
t0 = time.time()
# Seed for random shuffling the data
np.random.seed(10)
# Path of dataset
datasetPATH = '/home/gryphonlab/Ioannis/Works/RML/'
# Sample rate of audio
video_rate = 30
# Number of video frames for one spectrogram
num_frames = 7
# Stride between consecutive video frames
stride_frame = 2
# List of sessions or/and speakers to leave out
sessions_out = ['s4']


########################## FUNCTIONS ###########################


### Function for creating folders ###
def create_folder(PATH):
    
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print ('Error: Creating directory' + PATH)


### Function to convert emotion name to class number ###
def convert_emotion(emotion):

    # Check emotion to transpose
    if emotion == 'ha':
        label = 0
    elif emotion == 'su':
        label = 1
    elif emotion == 'sa':
        label = 2
    elif emotion == 'fe':
        label = 3
    elif emotion == 'an':
        label = 4
    elif emotion == 'di':
        label = 5
    
    return label


### Function to read all data of dataset ###
def read_data(datasetPATH):

    # Lists to append total face, spectrogram paths and emotion labels
    facesPATH, labels = list(), list()

    speakers = os.listdir(datasetPATH + 'InputSpectrograms/')

    for speaker in speakers:
        
        sessionPATH = datasetPATH + 'InputSpectrograms/' + speaker + '/'
        languages = os.listdir(sessionPATH)
        
        if len(languages) < 7:
            
            for language in languages:
                
                if language != 'Thumbs.db':
                    
                    speakerPATH = sessionPATH + language + '/'
                    videos = os.listdir(speakerPATH)
                    
                    for vid in videos:
                        
                        # Convert emotion name to class label
                        label = convert_emotion(vid[:2])
                        
                        videoPATH = speakerPATH + vid + '/'
                        spectrograms = os.listdir(videoPATH)

                        for spectrogram in spectrograms:

                            spectrogramPATH = videoPATH + spectrogram
                            # Find center frame of current spectrogram
                            center_frame = int(round(float(spectrogram[2:-4])*video_rate))
                            # Path of center frame
                            centerframePATH = datasetPATH + 'InputFaces/' + speaker + '/' + language + '/' + vid + '/frame'+str(center_frame)+'.jpg'
                            # Check that frame exists
                            if os.path.isfile(centerframePATH):

                               # Index of current frame number in path string
                                i = int(centerframePATH.find('frame')+5)
                                # Center frame number
                                center = int(centerframePATH[i:-4])
                                # Run to find all desired frames of a spectrogram
                                for f in range(num_frames):
                        
                                    # Desired frame according to center frame and stride
                                    desired_frame = center+stride_frame*(f-int((num_frames-1)/2))
                                    # Path of current frame
                                    facePATH = centerframePATH[:i]+str(desired_frame)+'.jpg'
                                    # Check that frame exists
                                    if os.path.isfile(facePATH):

                                        # Append to faces path list
                                        facesPATH.append(facePATH)
                                        # Append emotion label
                                        labels.append(label)
                            else:

                                print('ERROR[0]: Frame '+centerframePATH+' not found..!')

        else:
                    
            videos = languages
            
            for vid in videos:

                # Convert emotion name to class label
                label = convert_emotion(vid[:2])

                videoPATH = sessionPATH + vid + '/'
                spectrograms = os.listdir(videoPATH)

                for spectrogram in spectrograms:

                    spectrogramPATH = videoPATH + spectrogram
                    # Find center frame of current spectrogram
                    center_frame = int(round(float(spectrogram[2:-4])*video_rate))
                    # Path of center frame
                    centerframePATH = datasetPATH + 'InputFaces/' + speaker + '/' + vid + '/frame'+str(center_frame)+'.jpg'
                    # Check that frame exists
                    if os.path.isfile(centerframePATH):

                        # Index of current frame number in path string
                        i = int(centerframePATH.find('frame')+5)
                        # Center frame number
                        center = int(centerframePATH[i:-4])
                        # Run to find all desired frames of a spectrogram
                        for f in range(num_frames):
                        
                            # Desired frame according to center frame and stride
                            desired_frame = center+stride_frame*(f-int((num_frames-1)/2))
                            # Path of current frame
                            facePATH = centerframePATH[:i]+str(desired_frame)+'.jpg'
                            # Check that frame exists
                            if os.path.isfile(facePATH):

                                # Append to faces path list
                                facesPATH.append(facePATH)
                                # Append emotion label
                                labels.append(label)
            
                            else:

                                # Else raise error
                                print('ERROR[2] Frame '+centerframePATH+' not found..!')

                    else:

                        print('ERROR[1]: Frame '+centerframePATH+' not found..!')

    # Concatenate to numpy array
    data = np.array([facesPATH, labels])
    data = data.T
    
    return data


### Function to split data according speakers to train, eval data ###
def split_data(data):

    # Shuffle data randomly accros one axis=0
    np.random.shuffle(data)
    # Lists to append training and evaluation data
    train_data, eval_data = list(), list()
    # Run through all dataset elements
    for i in range(np.shape(data)[0]):

        data_for_eval = False

        for session_out in sessions_out:

            # Check if any of the sessions in the sessions_out list match to current face path string
            if not data_for_eval and data[i,0].find(session_out) >= 0:

                # Add database element to evaluation data
                eval_data.append(data[i])
                data_for_eval = True
        
        if not data_for_eval:

            # Else add database element to training data
            train_data.append(data[i])

    # Convert lists to numpy arrays
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)

    return train_data, eval_data


### Function to unroll faces and labels to numpy arrays ###
def unroll_data(data):

    # Split columns of data matrix
    facesPATH = data.T[0]
    labels = data.T[1]
    
    return facesPATH, labels


############################# MAIN ############################


def main():

    data = read_data(datasetPATH)
    train_data, eval_data = split_data(data)
    print('Training elements: ' + str(np.shape(train_data)))
    print('Evaluation elements: ' + str(np.shape(eval_data)))

    train_faces, train_labels = unroll_data(train_data)
    eval_faces, eval_labels = unroll_data(eval_data)

    d1 = {'train_faces': train_faces, 'train_labels': train_labels}
    d2 = {'eval_faces': eval_faces, 'eval_labels': eval_labels}
    #d3 = {'pred_faces': pred_faces, 'pred_labels': pred_labels}

    create_folder(datasetPATH+'visual/')

    df1 = pd.DataFrame(data=d1)
    df2 = pd.DataFrame(data=d2)
    #df3 = pd.DataFrame(data=d3)
    df1.to_csv( datasetPATH + 'Core/visual/training_data.csv' )
    df2.to_csv( datasetPATH + 'Core/visual/evaluation_data.csv' )
    #df3.to_csv( datasetPATH + 'Core//visual/prediction_data.csv' )

    # Execution time
    print( 'Execution time of createInputcsv.py [sec]: ' + str(time.time() - t0) )


# Control runtime
if __name__ == '__main__':
    main()