# Face Recognition Attendance System

## Introduction

Face recognition is one of the most important biometric recognition techniques. It is relatively simple to set up and covers an extensive range of applications varying from surveillance to digital marketing.

## Objective

- To design a real-time face recognition system that identifies people across the **BITS Pilani university campus**. Live video footage is provided as an input through the **100+** CCTV cameras installed at various vital locations across the campus. 
- To design a **web portal** that can *recognize* people by their faces, *mark their attendance*, record their *entry*, and *exit* time.

## Dataset

We use the data set obtained from *Student Welfare Division (SWD), BITS Pilani*. It contains information like `photo`, `ID No.`, `name` and the `hostel` of all the **5000+** students registered at the Pilani campus from the academic year `2014 to 2018`.

## Proposed Technique

Our algorithm is divided into three main parts:
1. Preprocessing
2. Model training
3. Web application development

### Phase I - Preprocessing

- The live video input from the CCTV camera is divided into frames at the rate of `30` frames per sec.

#### STEP 1: Face detection

![Outline_CNN_Model](./Images/Outline_CNN_Model.PNG?raw=true "Outline_CNN_Model")

We use two methods to detect faces in each video frame:
  - The first method is using *Convolutional Neural Networks (CNNs)* for face detection. It generates a `bounding box` around the face. 
  - To further increase the accuracy, we detect the face using a *HOG (Histogram of Gradients)* based detector.
  
![HOG](./Images/HOG.PNG?raw=true "HOG")

#### STEP 2: Facial Landmark points detection

Sixty-eight landmark points are identified on the face using the `Dlib` python library. These landmark points are crucial for the next phase.

![Dlib_Landmark_Points](./Images/Dlib_Landmark_Points.PNG?raw=true "Dlib_Landmark_Points")

#### STEP 3: Alignment

- The goal is to warp and transform the input image (coordinates) onto an output coordinate space, such that, each face in the output coordinate space: 
  1. Be *centred* in the image. 
  2. Be *rotated* such that the *eyes lie on a horizontal line* (i.e., the face is rotated such that the eyes lie along the same y-coordinates). 
  3. Be *scaled* such that the size of the faces is approximately identical.

- The above recognized facial landmark points are used for alignment purposes. We perform *affine transformation* using the above-recognised landmark points. Facial recognition algorithms perform better on aligned faces.

###  PHASE II - Model

- Our model proposes a solution that uses **only one image per individual** to detect the identity. 
- **128-dimensional embeddings** are generated for each of the images using the entire data pre-processing step which is then fed into the SVM for training.
- The model is trained using a **triplet loss function**. 
- Finally, after training the model and generating the embeddings, we recognize different faces using a `Support Vector Machine (SVM)` based classifier. 
- A test image’s embeddings are generated in the above fashion and finally compared with the other embeddings for successful classification.

![Triplet_Loss_Function](./Images/Triplet_Loss_Function.PNG?raw=true "Triplet_Loss_Function")

### PHASE III - Web Application Development

We built a web platform where students can register themselves and mark their attendance once registered.

![Home_Page](./Images/Home_Page.PNG?raw=true "Home_Page")

![Registration_Page](./Images/Registration_Page.PNG?raw=true "Registration_Page")


## Error Analysis

The possible reasons for the errors could be: 
1. Change in the person’s face over time - considerable facial change from the photo used in training. 
2. Two or more similar looking people - If there are multiple people with similar faces, then the model may wrongly classify a person as someone else. 
3. Lack of training data - Deep learning networks are known to increase their accuracy in increasing the data. Since we have only one image per person, therefore there is scope for the model to be trained more efficiently.


## Instructions to run

- Clone the project and download the source code.
- Once inside the folder, run the following command: 
    `python3 app.py`
- Go to: `localhost:5000` in your browser
- First Register yourself by clicking on `Register Image`
- Mark your attendance by clicking on `Start Surveillance Image`
