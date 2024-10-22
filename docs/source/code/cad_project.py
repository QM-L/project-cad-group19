"""
Project code for CAD topics.
----------------------------------------
This file contains the bulk of the surface code used to apply our methods and obtain our results.
Some functions/classes in this file mirror those in course-given files, with some minor changes or in some cases
no change. Though code-wise inefficient, this has been done on purpose, as to be as transparent as possible about 
what code provides the core structure of the program.
"""

import random
import numpy as np
import cad_util as util
import matplotlib
import matplotlib.pyplot as plt
import cad
import scipy
import scipy.io
from IPython.display import display, clear_output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

def nuclei_measurement():

    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"]  # shape (24, 24, 3, 20730)
    test_y = mat["test_y"]  # shape (20730, 1)
    training_images = mat["training_images"]  # shape (24, 24, 3, 21910)
    training_y = mat["training_y"]  # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n]  # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:]  # Get the 300 largest

    # Visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:, :, :, sort_ix_low.ravel()]
    X_large = training_images[:, :, :, sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # Dataset preparation
    imageSize = training_images.shape

    # Every pixel is a feature, so the number of features is height x width x color channels
    numFeatures = imageSize[0] * imageSize[1] * imageSize[2]
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    ## Train a linear regression model on the full training dataset
    model_full = LinearRegression()
    model_full.fit(training_x, training_y.ravel())  # Fit model to full training data

    # Predict the nuclei areas for the test set
    predicted_y_full = model_full.predict(test_x)

    ## Train a linear regression model using a reduced dataset (every 4th training sample)
    reduced_indices = np.arange(0, training_x.shape[0], 4)  # Select every 4th sample
    training_x_reduced = training_x[reduced_indices]
    training_y_reduced = training_y[reduced_indices]

    model_reduced = LinearRegression()
    model_reduced.fit(training_x_reduced, training_y_reduced.ravel())  # Fit model to reduced training data

    # Predict the nuclei areas for the test set with the reduced model
    predicted_y_reduced = model_reduced.predict(test_x)

    # Calculate and print errors
    mse_full = mean_squared_error(test_y, predicted_y_full)
    mae_full = mean_absolute_error(test_y, predicted_y_full)
    
    mse_reduced = mean_squared_error(test_y, predicted_y_reduced)
    mae_reduced = mean_absolute_error(test_y, predicted_y_reduced)
    
    print(f"Full sample model - MSE: {mse_full:.4f}, MAE: {mae_full:.4f}")
    print(f"Reduced sample model - MSE: {mse_reduced:.4f}, MAE: {mae_reduced:.4f}")

    # Visualize the results
    fig2 = plt.figure(figsize=(16, 8))
    
    ax1 = fig2.add_subplot(121)
    ax1.plot(test_y, predicted_y_full, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Actual Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    ax2 = fig2.add_subplot(122)
    ax2.plot(test_y, predicted_y_reduced, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Actual Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')

    plt.show()
    
def nuclei_classification(use_PCA=False, plot=True,static_params=[],results=True):
    ## dataset preparation
    
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images = mat["test_images"]  # (24, 24, 3, 20730)
    test_y = mat["test_y"]  # (20730, 1)
    training_images = mat["training_images"]  # (24, 24, 3, 14607)
    training_y = mat["training_y"]  # (14607, 1)
    validation_images = mat["validation_images"]  # (24, 24, 3, 7303)
    validation_y = mat["validation_y"]  # (7303, 1)

    ## dataset preparation
    training_x, validation_x, test_x = util.reshape_and_normalize(training_images, validation_images, test_images)
    r, c = training_x.shape
    
    # Transform dataset to PCA if called for
    if use_PCA:
        # Flatten the images for PCA
        num_train_samples,_ = training_x.shape
        num_val_samples,_ = validation_x.shape
        num_test_samples,_ = test_x.shape

        # Define flattened
        training_x_flat = training_x.reshape(num_train_samples, -1)
        validation_x_flat = validation_x.reshape(num_val_samples, -1)
        test_x_flat = test_x.reshape(num_test_samples, -1)

        # Apply PCA
        pca = PCA(n_components=0.95)  # Keep 95% of the explained variance
        pca.fit(training_x_flat)

        # Provide information about the PCA transformation
        explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
        num_components = pca.components_.shape[0]
        print(f"{explained:.1f}% explained using {num_components} components.")

        # Transform datasets
        training_x = pca.transform(training_x_flat)
        validation_x = pca.transform(validation_x_flat)
        test_x = pca.transform(test_x_flat)

        # redefine c
        r, c = training_x.shape

    ## Number of iterations
    num_iterations = 30 if (static_params == []) else 300
    
    #Define values for plotting
    xx = np.arange(num_iterations)
    training_loss = np.empty(*xx.shape)
    training_loss[:] = np.nan
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    g = np.empty(*xx.shape)
    g[:] = np.nan
    
    # Define ranges for mu and batch_size
    mu_range = [0.0001, 0.01]  # Learning rates
    batch_size_range = [32, 256]  # Batch sizes

    # Number of random samples to try for search. if set params are given, set searches to 1
    num_random_searches = 100 if (static_params == []) else 1

    best_mu = None
    best_batch_size = None
    best_val_loss = float('inf')  # Initialize to a large value

    # Random search over combinations of mu and batch_size
    for _ in range(num_random_searches):
        if (static_params == []):
            # Randomly sample hyperparameters
            mu = random.uniform(*mu_range)
            batch_size = random.randint(*batch_size_range)
        else:
            # Set to given parameters
            mu = static_params[0]
            batch_size = static_params[1]

        print(f"Testing mu={mu}, batch_size={batch_size}")
        
        # Initialize loss arrays for the current random search
        training_loss = np.full(xx.shape, np.nan)
        validation_loss = np.full(xx.shape, np.nan)
        
        # Re-initialize Theta for each run
        Theta = 0.02 * np.random.rand(c + 1, 1)

        # Training loop
        for k in np.arange(num_iterations):
            # pick a batch at random
            idx = np.random.randint(training_x.shape[0], size=batch_size)

            training_x_ones = util.addones(training_x[idx, :])
            validation_x_ones = util.addones(validation_x)

            # The loss function for this particular batch
            loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

            # Gradient descent update
            Theta_new = Theta - mu * cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

            # Record training and validation loss
            training_loss[k] = loss_fun(Theta) / batch_size
            validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta) / validation_x.shape[0]
    
            Theta = None
            Theta = np.array(Theta_new)
            Theta_new = None
            tmp = None
        
        # Final validation loss after training
        final_val_loss = validation_loss[-1]
        
        final_training_loss = training_loss[-1]

        # Update best hyperparameters if this run is better
        if final_val_loss < best_val_loss:
            best_mu = mu
            best_batch_size = batch_size
            best_val_loss = final_val_loss
            best_training_loss = final_training_loss

        if plot:
            # Plot training vs validation loss for each run
            fig2 = plt.figure(figsize=(7, 8))
            ax3 = fig2.add_subplot(111)
            q1, = ax3.plot(training_loss, label='Training Loss')
            q2, = ax3.plot(validation_loss, label='Validation Loss')
            ax3.set_title(f'mu={mu}, batch_size={batch_size}')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Loss')
            ax3.legend()
            #display(fig2)
    
    # Print the best combination of hyperparameters
    print(f"Best hyperparameters: mu={best_mu}, batch_size={best_batch_size}")
    print(f"Best validation loss: {best_val_loss}")
    
    if (static_params == []):
        # return best parameters if parameter search was performed
        return best_mu, best_batch_size
    
    # Return if no results are requested
    if not results:
        return None
    
    # if static params were given, pass the model on the test batch and return the accuracy
    test_x_ones = util.addones(test_x)
    pred_y = cad.sigmoid(np.array(test_x_ones.dot(Theta)))

    # calc accuracy
    test_accuracy = (test_y == np.round(pred_y)).sum()/(test_y.shape[0])
    print('Test accuracy: {:.2f}'.format(test_accuracy))

    # Plot final test predictions
    large_list = pred_y[(test_y==1).flatten()]
    small_list = pred_y[(test_y==0).flatten()]
    plt.figure()
    plt.hist(small_list, 50, alpha = 0.5)
    plt.hist(large_list, 50, alpha = 0.5)
    plt.legend(['Small (label = 0)','Large (label = 1)'], loc = 'upper center')
    plt.xlabel('Prediction')
    plt.title('Final test set predictions')
    plt.show()

    # Plot confusion matrix
    true_large_pred_large = len([num for num in large_list if num > 0.5])
    true_large_pred_small = len([num for num in large_list if num <= 0.5])
    true_small_pred_large = len([num for num in small_list if num > 0.5])
    true_small_pred_small = len([num for num in small_list if num <= 0.5])
    title = "Confusion Matrix PCA+LG" if use_PCA else "Confusion Matrix LG"
    plot_confusion_matrix(true_large_pred_large,true_small_pred_large,true_small_pred_small,true_large_pred_small,title)

def KNearestOptimized(k_values=[1, 3, 5, 7, 9], num_trials=10, batch_size=300):
    """
    Runs K-Nearest Neighbors classification on nuclei data using principal components up to
    95% explained variance,
    and selects the best K based on validation loss over random validation batches.
    
    Parameters:
    k_values: list, different values of K to test
    num_trials: int, number of random validation samples to test
    batch_size: int, number of random validation samples for each trial
    
    Returns:
    best_k: int, the value of K that minimizes validation loss
    best_val_loss: float, the corresponding validation loss
    """
    
    # Load data from the .mat file
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)
    
    # Extract test and training images, and labels
    test_images = mat["test_images"]
    test_y = mat.get("test_y")
    training_images = mat["training_images"]
    training_y = mat["training_y"]
    validation_images = mat["validation_images"]
    validation_y = mat["validation_y"]

    # Flatten and normalize the data
    training_x, validation_x, test_x = util.reshape_and_normalize(training_images, validation_images, test_images)
    
    # Find indices of small (0) and large (1) nuclei
    sort_ix_low = np.flatnonzero(training_y.flatten() == 0)[:700]  # Indices of the first 300 small nuclei
    sort_ix_high = np.flatnonzero(training_y.flatten() == 1)[:700]  # Indices of the first 300 large nuclei

    # Combine selected indices for labeled data
    labeled_data_indices = np.concatenate((sort_ix_low, sort_ix_high))

    # Apply PCA to reduce dimensionality (retain 95% of the variance)
    pca = PCA(n_components=0.95)
    training_x_pca = pca.fit_transform(training_x)
    validation_x_pca = pca.transform(validation_x)
    test_x_pca = pca.transform(test_x)  # Apply PCA to the test set
    
    # Filter the PCA data and labels for only the selected small and large nuclei
    labeled_training_x_pca = training_x_pca[labeled_data_indices]  # Only the 600 labeled samples
    labeled_labels = training_y[labeled_data_indices]  # Only the labels of the 600 samples

    # Combine PCA-transformed data with labels for only the labeled nuclei
    training_x_combined = np.hstack((labeled_training_x_pca, labeled_labels.reshape(-1, 1)))

    # Initialize variables for tracking the best k
    best_k = None
    best_val_loss = float('inf')
    best_val_predictions = None

    # Track the loss for each k value
    k_losses = []

    # Generalized Euclidean distance function (for n dimensions)
    def distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    # KNN function that works with any number of PCA components
    def KNN(dataset, image, k):
        
        distances = []
        for data_point in dataset:
         
            dist = distance(image, data_point[:-1])  # Compare using all PCA components
            distances.append((dist, data_point[-1]))  # Store distance and label
        
        # Sort by distance and select the k nearest neighbors

        distances = sorted(distances, key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # Count the number of small and large nuclei in the nearest neighbors
        count_small = sum(1 for d in k_nearest if d[1] == 0)
        count_big = sum(1 for d in k_nearest if d[1] == 1)
        # Assign label based on the majority vote
        return 0 if count_small > count_big else 1
    
    # Iterate over different k values
    print(f"Labeled data shape: {training_x_combined.shape}")
    for k in k_values:
        print(f"Testing K = {k}")
        trial_losses = []  # To store the validation loss for each trial
        
        # Repeat for multiple random trials
        for trial in range(num_trials):
            # Randomly select a batch of validation samples
            random_indices = random.sample(range(validation_x_pca.shape[0]), batch_size)
            val_images_batch = validation_x_pca[random_indices]
            val_labels_batch = validation_y[random_indices]

            # List to store predictions for the validation batch
            val_predictions = []
            
            # Run KNN on the selected validation batch
            for val_image in val_images_batch:

                val_image_with_label = val_image
                # Predict the label using KNN
                predicted_label = KNN(training_x_combined, val_image_with_label, k)
                val_predictions.append(predicted_label)
            
            # Convert the predicted labels into a numpy array
            val_predictions = np.array(val_predictions).reshape(-1, 1)
            
            # Calculate the validation loss (Mean Squared Error)
            val_loss = np.mean((val_predictions - val_labels_batch) ** 2)
            trial_losses.append(val_loss)

        # Calculate the mean loss over all trials for this k
        mean_val_loss = np.mean(trial_losses)
        k_losses.append(mean_val_loss)

        print(f"Validation loss for K = {k}: {mean_val_loss}")
        
        # Update the best K and validation loss if this one is better
        if mean_val_loss < best_val_loss:
            best_k = k
            best_val_loss = mean_val_loss
            best_val_predictions = val_predictions

    # Plot validation loss vs K values
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, k_losses, marker='o', linestyle='-', color='b')
    plt.title("Validation Loss vs K values")
    plt.xlabel("K")
    plt.ylabel("Validation Loss")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    # Test accuracy calculation using the best K
    print(f"Best K: {best_k}, with validation loss: {best_val_loss}")
    
    test_predictions = []
    for test_image in test_x_pca:
        predicted_label = KNN(training_x_combined, test_image, best_k)
        test_predictions.append(predicted_label)
    
    # Convert test predictions to numpy array
    test_predictions = np.array(test_predictions).reshape(-1, 1)

    # Calculate test accuracy
    test_accuracy = np.mean(test_predictions == test_y)

    print(f"Test Accuracy with K = {best_k}: {test_accuracy}")
    
    # Plot final test predictions
    large_list = test_predictions[test_y==1]
    small_list = test_predictions[test_y==0]
    plt.figure()
    plt.hist(small_list, 50, alpha = 0.5)
    plt.hist(large_list, 50, alpha = 0.5)
    plt.legend(['Small (label = 0)','Large (label = 1)'], loc = 'upper center')
    plt.xlabel('Prediction')
    plt.title('Final test set predictions')
    plt.show()

    # Plot confusion matrix
    true_large_pred_large = len([num for num in large_list if num > 0.5])
    true_large_pred_small = len([num for num in large_list if num <= 0.5])
    true_small_pred_large = len([num for num in small_list if num > 0.5])
    true_small_pred_small = len([num for num in small_list if num <= 0.5])
    title = "Confusion Matrix PCA+kNN"
    plot_confusion_matrix(true_large_pred_large,true_small_pred_large,true_small_pred_small,true_large_pred_small,title)


    return best_k, best_val_loss, test_accuracy




## Edited version of 2.3's Training class (with editable parameters)
class Training:
    
    def data_preprocessing(self):
        self.use_PCA = False
        ## load dataset (images and labels y)
        fn = '../data/nuclei_data_classification.mat'
        mat = scipy.io.loadmat(fn)

        training_images = mat["training_images"]
        self.training_y = mat["training_y"]

        validation_images = mat["validation_images"]
        self.validation_y = mat["validation_y"]

        test_images = mat["test_images"]
        self.test_y = mat["test_y"]

        ## dataset preparation
        # Reshape matrices and normalize pixel values
        self.training_x, self.validation_x, self.test_x = util.reshape_and_normalize(training_images,validation_images, test_images)      

        # Visualize several training images classified as large or small
        #util.visualize_big_small_images(self.training_x, self.training_y, training_images.shape)
        #util.visualize_big_small_images(training_images, self.training_y)

    def data_preprocessing_PCA(self):
        self.use_PCA = True
        ## load dataset (images and labels y)
        fn = '../data/nuclei_data_classification.mat'
        mat = scipy.io.loadmat(fn)

        training_images = mat["training_images"]
        self.training_y = mat["training_y"]

        validation_images = mat["validation_images"]
        self.validation_y = mat["validation_y"]

        test_images = mat["test_images"]
        self.test_y = mat["test_y"]

        ## dataset preparation
        # Reshape matrices and normalize pixel values
        self.training_x, self.validation_x, self.test_x = util.reshape_and_normalize(training_images, validation_images, test_images)

        # Flatten the images for PCA
        num_train_samples,_ = self.training_x.shape
        num_val_samples,_ = self.validation_x.shape
        num_test_samples,_ = self.test_x.shape

        self.training_x_flat = self.training_x.reshape(num_train_samples, -1)
        self.validation_x_flat = self.validation_x.reshape(num_val_samples, -1)
        self.test_x_flat = self.test_x.reshape(num_test_samples, -1)

        # Apply PCA
        pca = PCA(n_components=0.95)  # Keep 95% of the explained variance
        pca.fit(self.training_x_flat)

        # Provide information about the PCA transformation
        explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
        num_components = pca.components_.shape[0]
        print(f"{explained:.1f}% explained using {num_components} components.")

        # Transform datasets
        self.training_x = pca.transform(self.training_x_flat)
        self.validation_x = pca.transform(self.validation_x_flat)
        self.test_x = pca.transform(self.test_x_flat)

    def define_shapes(self, learning_rate, batchsize, n_hidden_features):

        self.learning_rate = learning_rate
        self.batchsize = batchsize

        in_features = self.training_x.shape[1]
        out_features = 1 # Classification problem, so you want to obtain 1 value (a probability) per image

        # Define shapes of the weight matrices
        self.w1_shape = (in_features, n_hidden_features)
        self.w2_shape = (n_hidden_features , out_features)

        return {'w1_shape': self.w1_shape,
                'w2_shape': self.w2_shape}


    def launch_training(self, n_epochs,show_plots=False):
        
        # Define empty lists for saving training progress variables
        training_loss = []
        validation_loss = []
        Acc = []
        steps = []

        # randomly initialize model weights
        self.weights = util.init_model(self.w1_shape, self.w2_shape)
        acc_height = 0.8
        print('> Start training ...')

        # Train for n_epochs epochs
        for epoch in range(n_epochs): 

            # Shuffle training images every epoch
            training_x, training_y = util.shuffle_training_x(self.training_x, self.training_y)

            for batch_i in range(self.training_x.shape[0]//self.batchsize):

                ## sample images from this batch
                batch_x = training_x[self.batchsize*batch_i : self.batchsize*(batch_i+1)]
                batch_y = training_y[self.batchsize*batch_i : self.batchsize*(batch_i+1)]

                ## train on one batch
                # Forward pass
                hidden, output = self.forward(batch_x, self.weights)
                # Backward pass    
                self.weights = self.backward(batch_x, batch_y, output, hidden, self.weights)

                ## Save values of loss function for plot
                training_loss.append(util.loss(output, batch_y))
                steps.append(epoch + batch_i/(self.training_x.shape[0]//self.batchsize))

            ## Validation images trhough network
            # Forward pass only (no backward pass in inference phase!)
            _, val_output = self.forward(self.validation_x, self.weights)
            # Save validation loss
            val_loss = util.loss(val_output, self.validation_y)
            validation_loss.append(val_loss)
            accuracy = (self.validation_y == np.round(val_output)).sum()/(self.validation_y.shape[0])
            Acc.append(accuracy)

            # Plot loss function and accuracy of validation set
            if show_plots:

                acc_height = max([accuracy + 0.02,acc_height])

                clear_output(wait=True)
                fig, ax = plt.subplots(1,2, figsize=(15,5))
                ax[0].plot(steps,training_loss)
                ax[0].plot(range(1, len(validation_loss)+1), validation_loss, '.')
                ax[0].legend(['Training loss', 'Validation loss'])
                ax[0].set_title(f'Loss curves after {epoch+1}/{n_epochs} epochs')
                ax[0].set_ylabel('Loss'); ax[0].set_xlabel('epochs')
                ax[0].set_xlim([0, 100]); ax[0].set_ylim([0, max(training_loss)])
                ax[1].plot(Acc)
                ax[1].set_title(f'Validation accuracy after {epoch+1}/{n_epochs} epochs')
                ax[1].set_ylabel('Accuracy'); ax[1].set_xlabel('epochs')
                ax[1].set_xlim([0, 100]); ax[1].set_ylim([min(Acc),acc_height])
                plt.show()
        print(f'> Training finished with accuracy = {accuracy}')

        # return accuracy to use elsewhere
        return accuracy
            
    def pass_on_test_set(self):
        
        # Forward pass on test set
        _, test_output = self.forward(self.test_x, self.weights)
        test_accuracy = (self.test_y == np.round(test_output)).sum()/(self.test_y.shape[0])
        print('Test accuracy: {:.2f}'.format(test_accuracy))

        # Plot final test predictions
        large_list = test_output[self.test_y==1]
        small_list = test_output[self.test_y==0]
        plt.figure()
        plt.hist(small_list, 50, alpha = 0.5)
        plt.hist(large_list, 50, alpha = 0.5)
        plt.legend(['Small (label = 0)','Large (label = 1)'], loc = 'upper center')
        plt.xlabel('Prediction')
        plt.title('Final test set predictions')
        plt.show()

        # Plot confusion matrix
        true_large_pred_large = len([num for num in large_list if num > 0.5])
        true_large_pred_small = len([num for num in large_list if num <= 0.5])
        true_small_pred_large = len([num for num in small_list if num > 0.5])
        true_small_pred_small = len([num for num in small_list if num <= 0.5])
        title = "Confusion Matrix PCA+cNN" if self.use_PCA else "Confusion Matrix cNN"
        plot_confusion_matrix(true_large_pred_large,true_small_pred_large,true_small_pred_small,true_large_pred_small,title)
        return test_accuracy

    def forward(self, x, weights):
        w1 = weights['w1']
        w2 = weights['w2']

        hidden = util.sigmoid(np.dot(x, w1))
        output = util.sigmoid(np.dot(hidden, w2))

        return hidden, output

    def backward(self, x, y, output, hidden, weights):
        w1 = weights['w1']
        w2 = weights['w2']

        # Caluclate the derivative with the use of the chain rule  
        dL_dw2 = np.dot(hidden.T, (2*(output - y) * util.sigmoid_derivative(output)))
        dL_dw1 = np.dot(x.T,  (np.dot(2*(output - y) * util.sigmoid_derivative(output), w2.T) * util.sigmoid_derivative(hidden)))

        # update the weights with the derivative (slope) of the loss function   
        w1 = w1 - self.learning_rate*dL_dw1
        w2 = w2 - self.learning_rate*dL_dw2
        return {'w1': w1,
                'w2': w2}


    def get_hyperparameters(self, n_trials=30, n_epochs=10):
        # parameter ranges
        learning_rate_range = [0.001, 0.05]
        batch_size_range = [128, 512]
        hidden_features_range = [500, 2000]

        best_acc = 0
        print("> Start random hyperparametric search ...")

        run = 0
        best_params = []
        for _ in range(n_trials):
            lr = random.uniform(*learning_rate_range)
            batch = random.randint(*batch_size_range)
            hidden_features = random.randint(*hidden_features_range)

            print(f'Testing lr={lr:.5f}, batch={batch}, hidden_features={hidden_features}')
            
            # Define the model with current set of hyperparameters
            self.define_shapes(lr, batch, hidden_features)

            # Train the model for a small number of epochs so it doesn't take ages
            accuracy = self.launch_training(n_epochs)
            run += 1
            print(f"(Trial {run}/{n_trials} finished)")
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = {lr, batch, hidden_features}

        print(f'Best parameters (lr, batch, features): {best_params}, with validation accuracy: {best_acc}')
        return best_params

def plot_confusion_matrix(true_positives=0,false_positives=0,true_negatives=0,false_negatives=0, title="insert a title"):
    matrix = np.array([[false_negatives,true_positives],
                       [true_negatives,false_positives]])

    fig, ax = plt.subplots()
    im = ax.imshow(matrix,cmap="Reds")
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap="Reds"), ax=ax)
    
    labels = [["small","big"],["big","small"]]
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(2), labels=labels[0])
    ax.set_yticks(np.arange(2), labels=labels[1])
    ax.set_ylabel("Labeled")
    ax.set_xlabel("Predicted")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            COLOR = "w" if value > 0.5*max(matrix.flatten()) else "k"
            text = ax.text(j, i, value, ha="center", va="center", color=COLOR)
    # Loop over data dimensions and create text annotations.

    ax.set_title(title)
    fig.tight_layout()
    plt.show()