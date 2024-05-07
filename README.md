# License Plate Recognition System

This project is designed to automatically detect and recognize license plates from vehicle images. It utilizes a combination of image processing techniques and a convolutional neural network (CNN) for character recognition. The end-to-end process involves extracting the number plate, segmenting the characters, training a model to recognize these characters, and predicting the plate number.

## Overview

The system performs the following tasks:

1. **Extracting the Number Plate**: Detect the number plate from a vehicle image using OpenCV.
2. **Character Segmentation**: Identify individual characters on the plate by finding contours and drawing rectangles around each character.
3. **Model Training**: A CNN model is trained on a dataset containing 36 classes ('A'-'Z' and '0'-'9').
4. **Character Recognition**: The trained model predicts the characters from the segmented license plate and the results are compared to the actual characters on the plate.

### Built With

- **Keras**: For designing and training the CNN model.
- **TensorFlow**: Backend engine for Keras.
- **Matplotlib**: For generating images of the number plates and character segments for visualization.
- **OpenCV**: Used for image processing tasks such as contour detection.
- **NumPy**: For handling high-level mathematical functions and array operations.
- **Pandas**: Used occasionally for data manipulation and analysis, though its use is minor in this project.

### Result

The system achieved 94% accuracy on the test dataset, demonstrating its effectiveness in recognizing varied license plates under different conditions.

## Streamlit Interface

To make the license plate recognition system more accessible and user-friendly, a Streamlit web interface has been integrated. Users can upload images of vehicles, and the system will display the detected number plate along with the recognized characters.

### Features of the Streamlit Interface:

- **Image Upload**: Users can upload images in JPEG, PNG, or JPG format.
- **Real-time Results Display**: After processing the uploaded image, the interface displays the original image, the image with the detected license plate highlighted, and the recognized characters.
- **Responsive Feedback**: The interface provides feedback on the processing stages and outputs the recognized number plate text.

## Getting Started

To run this project locally, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python installed, and set up a virtual environment.
3. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit application:

    ```bash
    streamlit run prediction.py
    ```

5. Navigate to `http://localhost:8501` in your web browser to interact with the application.

## Conclusion

This License Plate Recognition System showcases the power of machine learning and image processing to solve real-world problems in an interactive and user-friendly manner. The addition of the Streamlit interface allows for easy demonstration and real-time use of the system.

Feel free to contribute to the project by submitting pull requests or suggesting enhancements through the issues tab.



## Demo
![number_plate_recognition](https://user-images.githubusercontent.com/53619729/162380150-1747663c-5105-472b-b1ed-be76c6cc9ad1.gif)




## Creators

* V Manikanta Sanjay 
* Sri Hari KR. ➡️ [LinkedIn](https://www.linkedin.com/in/sri-hari-k-r-6b04a7176/) 
* Suyog P. ➡️ [LinkedIn](https://www.linkedin.com/in/suyog-p-a85aa51aa/)
## License

[MIT](https://github.com/ManikantaSanjay/indian_number_plate_character_recognition/blob/master/LICENSE)
