# lip-gesture-recognition
Classifying lip gestures in real-time using Python. The source code is able to differentiate among the following gestures : [Mouth -open, Mouth-close, Smile, Lip-folding inwards, and kissing lips]. 

To use:
1. Clone this repository.
2. Download dlib landmark file from here : https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat
3. Add the file in the same folder as python file.
4. Run the python file from the preferred IDE.

#Using DNN module
NOTE : Currently supports only mouth open and close gestures.

To use:
1. Download the 'using_dnn' folder.
2. Create folders src/ and model/ inside that folder.
3. Download dlib landmark file from here : https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat and add this file to model/ folder.
4. In the jupyter notebook, change the absolute path for the caffemodel and prototextpath as per your system's file hierarchy.
