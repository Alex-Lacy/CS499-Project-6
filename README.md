Our Neural Network comparison algorithm can be found here: [NNet Convolutional vs Connected](https://github.com/Alex-Lacy/CS499-Project-6/blob/master/Project%206.py)

This time, our algorithm and driver code are packaged into the same file.  Additionally, we rely on libraries instead of our own from-scratch NN algorithm.  You will have to use TensorFlow in order for it to run properly.  If you are using an IDE, such as PyCharm, then installing all required packages (see imports section of code) is easily acchieved through the settings menu (if it doesn't automatically)

Additionally, this time our code tests different parameters for optimization purposes.

Our code is tested to only work on numerical classification data.  For our results, we used [Stanford's Train Zip](https://web.stanford.edu/~hastie/ElemStatLearn/data.html) for our data. 

In order to use our code effecitvely, please be sure you have the data file in the same directory as the driver function, and be sure to import it in the code.

Our code automatically finds the optimal number of epochs and uses them.  It outputs graphs for similar convolutional neural networks and fully connected neural networks.  Any additional testing would need added by a user.

Good luck!
