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
       * Here basically the workflow is:
           ** Original dataset -> map() -> shuffle() -> batch() -> prefetch() -> PrefetchDataset .
       * map() is used map predefined function to a dataset (the predefined function is preprocess_img).
       * shuffle() is used to shuffle as the name suggests.
       * batch() to make the dataset into batches.
       * prefetch() prepares subsequent batches of data whilst other batches of data are being computed on (improves data loading speed but costs memory).
       * Note: Can't batch tensors of different shapes (e.g. different image sizes, need to reshape images first, hence our preprocess_img() function)
       * num_parallel_calls=tf.data.AUTOTUNE will parallelize preprocessing and significantly improve speed.
       * <img width="529" alt="image" src="https://github.com/Mokshith1708/FoodVision_CV/assets/146767844/616dca3f-8875-40b7-81c6-59186d30dc8b">
### d) Callbacks:
       * Here tf.keras.callbacks.ModelCheckpoint() is used.
       * This is used to save the model's progress at different intervals so that we can load it and reuse it.
       * Here we are monitoring the val_accuracy.
### e) Mixed pecision Training:
       * Concept of Mixed Precision Training: Mixed precision training leverages both float32 (single-precision, 32-bit) and float16 (half-precision, 16-bit) data types to 
                                              optimize GPU memory usage. By using float16 where high precision is not critical, more data can be processed simultaneously, 
                                              enhancing computational efficiency.

       * Benefits and Performance: This technique significantly improves performance on modern GPUs (compute capability 7.0+), potentially accelerating training speeds by up 
                                   to 3x. The reduction in memory usage allows for larger models or batch sizes, leading to more efficient utilization of GPU resources.

       * Implementation in TensorFlow: TensorFlow supports mixed precision training, enabling models to automatically balance the use of float32 and float16 data types. This 
                                       optimization is particularly beneficial in deep learning tasks, providing faster training times and improved resource management 
                                       without compromising model accuracy.

       *  Note: If your GPU doesn't have a score of over 7.0+ (e.g. P100 in Google Colab), mixed precision won't work
       *  mixed_precision.set_global_policy(policy="mixed_float16") is used to achieve it.
### f) Feature Extraction Model:
       * Basically there are mainly 5 major layers in this model:
            a) Input layer
            b) EfficientNetB0 Layer
            c) Pooling Layer
            d) Dense layer
            e) Softmax layer (basically output layer)
       * Input layer is created with input_shape: (224,224,3)
       * EfficientNet-b0 is a convolutional neural network that is trained on more than a million images from the ImageNet database [1].
       * The pooling layer, specifically GlobalAveragePooling2D, aggregates spatial information from a feature map by computing the average value of each feature map, 
         resulting in a reduced spatial dimensionality while retaining important features.
       * The dense layer, also known as a fully connected layer, connects every neuron in the previous layer to every neuron in the current layer, applying weights to the 
         input data and producing an output. This layer is crucial for learning complex patterns in data during training in neural networks.
       * The softmax layer is often used as the output layer in classification tasks because it transforms the raw predictions (logits) of a model into probabilities. These 
         probabilities indicate the likelihood of each class being the correct classification.
       * Here all the layer except EfficientNetB0 are trainable.
       * So by running this model all the custom layers gets trained except the EfficientNetB0 layers (this is fetature extraction model)
       * This model generally attained the accuracy of 72.8%
 ### g) Few changes:
       * EarlyStopping: Prevents the model from training further when it stops improving on the validation set, saving computational resources and preventing overfitting.
       * ModelCheckpoint: Ensures that the best model so far is saved based on val_loss, allowing you to later use this model for evaluation or deployment.
       * ReduceLROnPlateau: Adjusts the learning rate dynamically to help the model converge more effectively. As the model gets closer to optimal performance, smaller 
         learning rate adjustments prevent overshooting and stabilize training.
       * EarlyStopping Callback: Monitors validation loss (val_loss) and stops training if val_loss doesn't improve for 5 consecutive epochs (patience=3). This prevents 
                                 overfitting and saves training time.
       * ModelCheckpoint Callback: Saves the best performing model during training based on val_loss. This ensures that you have the best model saved even if training stops 
                                   early due to EarlyStopping.
       * ReduceLROnPlateau Callback: Adjusts the learning rate (lr) when validation loss (val_loss) plateaus, reducing it by a factor of 0.2 (factor=0.2) if val_loss doesn't 
                                     improve for 2 epochs (patience=2). This helps in fine-tuning the model as it gets closer to convergence.
 ### h) Fine-tuned Model:
       * Now slowly the few layers of efficientNetB0 are unfrozen and exposed to training.
       * Actually as our data set is very big all the layers were unforzen and were exposed to training.
       * Actually the triaining was set for only 10 epochs. Actually it is preffered to set it for 100 epochs and check, but as it takes lot of time and lack of resources 
         with me i trained it for 10 epochs.
       * The results were good. The final accuracy was 80.7%.

 ### Both the Feature extraction model and Fine tuned methods along with final weight metrices are attached above.
 ### What can be done to this:
    * Changing number of epochs and running the model .
    * Reducing the trainable layers.
    * Can show the output in more representable way.
##### This actually beats the result of DeepFood, a 2016 paper which used a Convolutional Neural Network trained for 2-3 days to achieve 77.4% top-1 accuracy.
##### Note: This was done as a part of course taken in udemy. So codes and workflow was learnt and taken form that. The course is TensorFlow for Deep Learning Bootcamp by Andrei Neagoie and Daniel Bourke
 
