from __future__ import unicode_literals, print_function, division, absolute_import
from bin.read_mnist import read_num

import numpy as np
import matplotlib.pyplot as plt

#################################
# Part to implement on your own #
#################################

def main():
    """
    This is the starting point of your code!
    You can do what you want here, but the general workflow is explained here.

    This only serves as a base for your operations. You can modify this function as much as you want.
    In addition, you can use other functions to help aid your along the way.

    1. "Train" your model by adding images element wise.
    So if you have a 5, just take all the other existing 5s and add them together.
    Simulanously, make a another array to keep track of how many times each number appeared.

    See: train() for more details (line 80)

    Optional Tip: It might make it easier if you save the trained model in a file somewhere.
    Training is the easier part to implement, and it takes the longest time to do.

    2. Create your model.

    2. "Predict" other sets by taking the total probability that it can be each number (from 0-9)
    2a. To take the total probability, we use a slightly different formula:

    P(N) = ((Similar Pixels in training) + 1) / ((Num of appearances in Training set) + 11)

    2b. Find the Maximum Probability to find the result.
    This technique is called Maximum Likelyhood

    See: predict() for more detail (line 90)
    """


    ###################################################################################################
    # The following code is sample code to get you aquainted with read_num() and generator functions. #
    # It's your choice if you want to use Numpy or just use regular Python lists.                     #
    ###################################################################################################

    print("Training Data set...")
    model, totals = train()
    print("Producing heatmap...")
    produce_heatmap(model, True, True)
    print("Testing results...")
    num_right, num_wrong = predict(model, totals, True)

    print("Accuracy: %.4f" % (float(num_right)/num_wrong))


def train():
    """
    Trains your model using read_num()

    @return: (model, total)
    """
    #######################################################################################
    # @TODO:                                                                              #
    # 1. Use read_num('training') to begin reading training data                          #
    # 2. Use a for loop to iterate through the generator.                                 #
    # 3. Add the model indexed at the resultant number and the training set element wise. #
    #                                                                                     #
    # ex. Some number A is represented with np.array([1, 10, 10...., 0]).                 #
    # You should add this array element wise with model[A].                               #
    #                                                                                     #
    # 4. Increment the total.                                                             #
    #                                                                                     #
    # ex. The number A was the number represented with the array.                         #
    # So increment total[A] += 1                                                          #
    #######################################################################################

    # Store model here! (Optional)
    model = np.zeros([10, 28, 28])
    # store totals here!

    # This will accumulate the total
    totals = np.zeros(10)

    # Create a generator to begin reading numbers
    generator_func = read_num('training')

    for arr, num in generator_func:
        model[num] += arr
        totals[num] += 1

    # After you train your model, you may want to plot a heatmap of it
    # Run produce_heatmap(model, True, True) to save your plot as an image
    produce_heatmap(model, True, True)

    return model, totals


def predict(model, totals, v=False, vv=False, every=False, t_max=1000):
    """
    Predict your model using the MNIST database.

    @return: (number right, total number)
    """

    ###################################################################################################
    # @TODO:                                                                                          #
    # 1. Use read_num('test') to begin reading testing data                                           #
    # 2. Since it takes a while to go through all of the training data, I recommend iterating through #
    # a couple hundred (300 is pretty good) to begin with.                                            #
    # 3. For all the pixels in the number, do                                                         #
    #                                                                                                 #
    # P(N) = ((Similar Pixels in training) + 1) / ((Num of appearances in Training set) + 11)         #
    #                                                                                                 #
    # 4. Your end probability is each probability of the pixel mulitplied with each other.            #
    # 5. Obtain the index of the highest probability. np.argmax() may help you here.                  #
    # 6. The index you got is your prediction. Compare it with the answer.                            #
    #                                                                                                 #
    # This is probably the hardest part. Ask the Instructor for help!                                 #
    ###################################################################################################

    num_total = 0
    num_right = 0

    pred = process(model, totals, vv)

    for p, n in pred:
        if p == n:
            num_right += 1

        num_total += 1

        if v:
            if num_total == 50:
                print("%6d" % (num_total))
            elif num_total % 50 == 0:
                print("\r%6d" % (num_total))

        if (not every) and num_total == t_max:
            print("\rDone    ")
            return num_right, num_total

    return num_right, num_total


def process(model, totals, v=False):
    """
    Processes the model.

    This is the function where all of the probability calculations are happening.

    @yield: (predicted number, Actual Number)
    """

    test_set = read_num('test')

    prob = np.vectorize(map_prob)

    for arr, num in test_set:
        # Create the probability map
        prob_map = [np.multiply.reduce(
            np.multiply.reduce(
                prob(arr, model[i], totals[i])))
                    for i in range(10)]

        predict = np.argmax(prob_map)

        if v:
            if predict != num:
                pprint_num(arr)

        yield predict, num


def map_prob(test_elm, model_elm, total):
    """
    Function that calculates the probability (element wise)

    This is a function that is vectorized. You will most likely implement this
    as a for loop instead.
    """

    outof = model_elm

    if test_elm == 0:
        outof = total - outof

    return (np.float128(outof) + 1) / (total + 11)

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

def produce_heatmap(model, every=True, save=False):

    #################################################################################
    # This function is a helper function for your needs.                            #
    # This will save a heatmap of your models.                                      #
    #                                                                               #
    # @args:                                                                        #
    # model - your created model                                                    #
    # save - save a figure as an image rather than displaying it                    #
    # every - pass in the model (rather than a single number) and display a heatmap #
    #################################################################################

    col_label = range(28)
    row_label = range(28)

    if every:
        for i in range(10):
            plt.pcolor(np.flipud(model[i]))
            plt.xticks(col_label)
            plt.yticks(row_label)
            plt.axis('off')
            plt.title("HeatMap for %d" % (i))
            cb = plt.colorbar()
            cb.set_label("Frequency")

            if save:
                plt.savefig('imgs/%d.png' % (i), bbox_inches='tight')
            else:
                plt.show()

            plt.close()

    else:
        plt.pcolor(np.flipud(model))
        plt.xticks(col_label)
        plt.yticks(row_label)
        plt.axis('off')
        cb = plt.colorbar()
        cb.set_label("Frequency")

        if save:
            plt.savefig('imgs/temp.png', bbox_inches='tight')
        else:
            plt.show()

        plt.close()

######################
# Internal Functions #
######################

if __name__ == '__main__':
    main()
