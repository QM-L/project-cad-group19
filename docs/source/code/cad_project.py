"""
Project code for CAD topics.
----------------------------------------
This file contains the bulk of the surface code used to apply our methods and obtain our results.
Some functions/classes in this file mirror those in course-given files, with some minor changes or in some cases
no change. Though code-wise inefficient, this has been done on purpose, as to be as transparent as possible about 
what code provides the core structure of the program.
"""

import numpy as np
import cad_util as util
import matplotlib
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io
import random

def nuclei_measurement():

    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # dataset preparation
    imageSize = training_images.shape
    
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    ## training linear regression model
    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model for measuring
    # the area of nuclei in microscopy images. Then, use the trained model
    # to predict the areas of the nuclei in the test dataset.
    #---------------------------------------------------------------------#

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1  = fig2.add_subplot(121)
    line1, = ax1.plot(test_y, predicted_y, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # TODO: Train a model with reduced dataset size (e.g. every fourth
    # training sample).
    #---------------------------------------------------------------------#

    # visualize the results
    ax2  = fig2.add_subplot(122)
    line2, = ax2.plot(test_y, predicted_y, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')


def nuclei_classification():
    ## dataset preparation
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images = mat["test_images"] # (24, 24, 3, 20730)
    test_y = mat["test_y"] # (20730, 1)
    training_images = mat["training_images"] # (24, 24, 3, 14607)
    training_y = mat["training_y"] # (14607, 1)
    validation_images = mat["validation_images"] # (24, 24, 3, 7303)
    validation_y = mat["validation_y"] # (7303, 1)

    ## dataset preparation
    training_x, validation_x, test_x = util.reshape_and_normalize(training_images, validation_images, test_images)      
    
    ## training linear regression model
    #-------------------------------------------------------------------#
    # TODO: Select values for the learning rate (mu), batch size
    # (batch_size) and number of iterations (num_iterations), as well as
    # initial values for the model parameters (Theta) that will result in
    # fast training of an accurate model for this classification problem.
    #-------------------------------------------------------------------#

    xx = np.arange(num_iterations)
    loss = np.empty(*xx.shape)
    loss[:] = np.nan
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    g = np.empty(*xx.shape)
    g[:] = np.nan

    fig = plt.figure(figsize=(8,8))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('mu = '+str(mu))
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    ax2.set_ylim(0, 0.7)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    for k in np.arange(num_iterations):
        # pick a batch at random
        idx = np.random.randint(training_x.shape[0], size=batch_size)

        training_x_ones = util.addones(training_x[idx,:])
        validation_x_ones = util.addones(validation_x)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

        # gradient descent
        # instead of the numerical gradient, we compute the gradient with
        # the analytical expression, which is much faster
        Theta_new = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

        loss[k] = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]

        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)

        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None

        display(fig)
        clear_output(wait = True)
        plt.pause(.005)

## Edited version of 2.3's Training class (with editable parameters)
class Training:
    
    def data_preprocessing(self):
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
        plot_confusion_matrix(true_large_pred_large,true_small_pred_large,true_small_pred_small,true_large_pred_small)

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