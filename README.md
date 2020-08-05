# indian_number_plate_character_recognition
Libraries used-
i>Keras
ii>Tensorflow
iii>Matplotlib
iv>OpenCV

This model contains 3 parts.First is extracting number plate from the image of the vehicle.Second,we find contours and try to draw rectangles around each character of the number plate.
Third,The model training is done on a dataset containing 36 classes,which are 'A'-'Z' and 0-9 using a CNN model architecture with softmax activation fucntion.Later,the model prediction is compared with the
characters in the number plate.Using 20 epochs i am obtaining around 94% accuracy.

CREATORS: V Manikanta Sanjay ; Sri Hari KR 
