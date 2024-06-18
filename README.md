# FoodVision_CV
## Descrition:
 Food Vision: A Transfer Learning Model built on the Food101 Dataset using some deep learning techniques.

## Where to find the code?
 All the code written is available in the FoodVision_code.ipynb.
 In this model mixed precision training was used.
 
 ## Note:
     1) Mixed precision training is avilable from TensorFlow 2.4.0 .
     2) To use mixed precision training we need to have GPU with compatibility score of 7.o +.
     3) This model was written and run in google colab as it offers T4 GPU.

 ## About the project:
  ### a) Datasets: 
         * The dataset Food101 is taken from tensorFlow datasets. [link](https://www.tensorflow.org/datasets/catalog/food10)
         * This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training 
           images.On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and 
           sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels (Download Size: 4.65Gib).
         * So there are a total of 75,750 training images and 25,250 testing images.
         * Tensorflow_datasets api was used to load the data.

### b) Preprocessing:
       * Initially the data was in uint8 which is convereted into float32 dtype (using tf.cast()).
       * All the images were of different size. They were converted to same size tensors using tf.image.resize().
       * All the pixels were normalized.
### c) Turning into batches and prefetching:
       * We turn our data from 101,000 image tensors and labels (train and test combined) into batches of 32 image and label pairs, thus enabling it to fit into the memory 
         of our GPU.
       * 
       
