# trabalho_graduacao

Neural Network project.


To load MNIST digit database you should run the Mnist_Loader.py. Remember to change the folder path.
Mnist_Loader.py (a) and Mnist_loader_digitos.py (b) are both doing almost the same thing, but in (a) the result is an 10-index array while (b) are just 0 or 1.

OCR.py is use to load a picture containg a bunch of digit. The result are all the digit in the picture in separate images, each image is 28x28 size.

Finally NeuralNetwork4.py is where the neural network run. It will first load the MNIST database, then start the training section and finishing with the images that you have saved after running the OCR.py.
