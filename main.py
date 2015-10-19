from __future__ import unicode_literals, print_function, division, absolute_import
from bin.read_mnist import read_num

import numpy as np
import matplotlib.pyplot as plt

#################################
# Part to implement on your own #
#################################

def main():
    #################################################################################################
    # This is the starting point of your code!                                                      #
    # You can do what you want here, but the general workflow is explained here.                    #
    #                                                                                               #
    # 1. "Train" your model by adding images element wise.                                          #
    # So if you have a 5, just take all the other existing 5s and add them together.                #
    # Simulanously, make a another array to keep track of how many times each number appeared.      #
    #                                                                                               #
    # Tip: It might make it easier if you save the trained model somewhere.                         #
    # Training is the easier part to implement, and it takes the longest time to do.                #
    #                                                                                               #
    # 2. Create your model.                                                                         #
    #                                                                                               #
    # 2. "Predict" other sets by taking the total probability that it can be each number (from 0-9) #
    # 2a. To take the total probability, we use a slightly different formula:                       #
    #                                                                                               #
    # P(N) = ((Similar Pixels in training) + 1) / ((Num of appearances in Training set) + 11)       #
    #                                                                                               #
    # 2b. Find the Maximum Probability to find the result.                                          #
    # This technique is called Maximum Likelyhood                                                   #
    #################################################################################################

    # This is the model used for processing.
    # 0 will be stored in the 0th index, 1 will be stored in the 1st index and so on
    model = np.zeros([10, 28, 28])

    # Example code
    # Create a generator to begin reading numbers
    generator_func = read_num('training')

    # You can call generator functions in two ways:
    # Method 1. Use the next() function.
    arr_and_num = generator_func.next()

    # Read_num is a tuple. The 0th index will return a numpy array
    arr_like = arr_and_num[0]
    # The 1st will return the number the image is supposed to represent
    num = arr_and_num[1]

    # Using the pretty print function to visualize the number
    pprint_num(arr_like)
    # Confirm the image is our number
    print(num)
    
    # Method 2. Using a for loop
    for arr_and_num in generator_func:
        # Won't go through everything, but the logic is the same.
        # For the first one there is 60000 data sets. So you probably
        # Don't want to print them all...
        break



####################################
# Utility Functions for you to use #
####################################

def pprint_num(arr_like):
    ########################################################################
    # This function is a helper function for your needs.                   #
    # This will print your number in such a way so that it will            #
    # be completely visible as a number, rather than an array form         #
    #                                                                      #
    # @args:                                                               #
    # arr_like - An array like object of an array representing the number. #
    ########################################################################

    dash = "-" * 28

    print(dash)
    for i in range(len(arr_like)):
        str_build = ""
        for j in range(len(arr_like[0])):
            str_build += "%d" % (arr_like[i][j])

        print(str_build)

    print(dash)

def produce_heatmap(model, save=False):
    
    ##############################################################
    # This function is a helper function for your needs.         #
    # This will save a heatmap of your models.                   #
    #                                                            #
    # @args:                                                     #
    # model - your created model                                 #
    # save - save a figure as an image rather than displaying it #
    ##############################################################
    
    col_label = range(28)
    row_label = range(28)

    for i in range(10):
        cb = plt.colorbar()
        cb.set_label("Frequency")
        plt.pcolor(model[i].flipud())
        plt.xticks(col_label)
        plt.xlabel('X')
        plt.yticks(row_label)
        plt.ylabel('Y')
        plt.title("HeatMap for %d" % (i))
        plt.show()
        plt.close()


######################
# Internal Functions #
######################

if __name__ == '__main__':
    main()
