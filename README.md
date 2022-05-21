<div align="Center">
<h1> Automatic Number Plate Recognition (ANPR) for Indian Vehicles </h1>
</div>

## About

This model involves performing 4 tasks.

- Extracting number plate from the image of the vehicle.
- Find contours and try to draw rectangles around each character of the number plate.
- The model training is done on a dataset containing 36 classes,which are 'A'-'Z' and 0-9 using a CNN model architecture.
- Finally, the model prediction is compared with the characters in the number plate.

## Python Libraries Used

- Keras
- Tensorflow
- Matplotlib
- OpenCV

 
## Result
94% accuracy was achieved on the test dataset.

## Demonstration


![number_plate_recognition](https://user-images.githubusercontent.com/53619729/162380150-1747663c-5105-472b-b1ed-be76c6cc9ad1.gif)




## Creators

* V Manikanta Sanjay 
* Sri Hari KR 

## License

[MIT](https://github.com/ManikantaSanjay/indian_number_plate_character_recognition/blob/master/LICENSE)
