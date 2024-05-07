import cv2
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import datetime

def load_image(path):
    return cv2.imread(path)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def extract_plate(img):
    # Convert the image to grayscale to help with the detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: Apply GaussianBlur to reduce noise and improve detection quality
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Load the trained cascade classifier
    plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')
    
    # Detect plates
    plate_rect = plate_cascade.detectMultiScale(blurred, scaleFactor=1.5, minNeighbors=5)
    
    plate_img = img.copy()
    
    for (x, y, w, h) in plate_rect:
        # Increase the margin a bit more for width (w)
        margin_w = int(0.1 * w)  # Increase width margin
        margin_h = int(0.1 * h)  # Increase height margin

        # Safeguard against clipping issues
        x_start = max(0, x - margin_w)
        y_start = max(0, y - margin_h)
        x_end = min(img.shape[1], x + w + margin_w)
        y_end = min(img.shape[0], y + h + margin_h)

        # Extract the plate
        plate = img[y_start:y_end, x_start:x_end]

        # Draw a rectangle around the plate
        cv2.rectangle(plate_img, (x_start, y_start), (x_end, y_end), (60, 51, 255), 3)

    return plate_img, plate

def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

#             Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)
            
    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img



def show_results(char):
    # Load the model from the specified path
    model = tf.keras.models.load_model('saved_model/my_model')
    print(model.summary())
    
    # Dictionary for character lookup
    dic = {i: c for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    
    # List to hold the predicted output characters
    output = []
    
    # Iterate over each character image in the input list 'char'
    for ch in char:
        # Resize the character image to the required input size for the model
        img_ = cv2.resize(ch, (28,28))
        
        # Fix the image dimension to match model input requirements
        img = fix_dimension(img_)
        
        # Reshape the image to match the input shape expected by the model: (1, 28, 28, 3)
        img = img.reshape(1, 28, 28, 3)
        
        # Predict the class probabilities for the input image
        probabilities = model.predict(img)
        
        # Find the index of the class with the highest probability
        y_ = np.argmax(probabilities, axis=1)[0]
        
        # Lookup the character corresponding to the predicted class index
        character = dic[y_]
        
        # Append the predicted character to the output list
        output.append(character)
    
    # Join all predicted characters into a single string
    return ''.join(output)

# def main():
#     plate_img2,plate1 = extract_plate(cv2.imread('swift_v2.jpeg'))
#     cv2.imwrite('car_plate.png',plate1)
#     img = load_image('car_plate.png')
#     char = segment_characters(img)
#     print(char)
#     plate_number = show_results(char)
#     print("Detected plate number:", plate_number)

# if __name__ == "__main__":
#     main()
# Your existing functions go here (make sure they are correctly defined in this script or imported)
# from your_module import extract_plate, segment_characters, show_results, fix_dimension



def main():
    st.title("License Plate Recognition System")
    st.write("Upload an image with a vehicle's license plate to detect and recognize the plate number.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the image with PIL
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        # Convert PIL image to NumPy array and change color from RGB to BGR
        image_np = np.array(image_pil)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process the image
        plate_img, plate = extract_plate(image_cv)

        if plate is not None:
            # Convert processed OpenCV image back to PIL to display in Streamlit
            plate_img_pil = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
            st.image(plate_img_pil, caption='Image with detected plate', use_column_width=True)

            # Run character segmentation and recognition
            char = segment_characters(plate)
            if char is not None and char.any():
                plate_number = show_results(char)
                st.write(f"Detected Plate Number: {plate_number}")
            else:
                st.write("No characters detected.")
        else:
            st.write("No plate detected in the image.")

if __name__ == "__main__":
    main()