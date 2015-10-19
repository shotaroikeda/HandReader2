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
    totals = np.zeros([10])

    # Example code
    # Create a generator to begin reading numbers
    generator_func = read_num('training')

    for arr, num in generator_func:
        model[num] += arr
        totals[num] += 1

    return (model, totals)


def predict(model, totals, v=False):
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


def process(model, totals, v=False, vv=False, every=False):
    pred = predict(model, totals, vv)

    total_right = 0
    total = 0
    for p, n in pred:
        if p == n:
            total_right += 1

        total += 1

        if v:
            if total % 50 == 0:
                print(total) 

        if (not every) and total == 1000:
            return total_right, total

    return total_right, total

def map_prob(test_elm, model_elm, total):
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
    model, totals = main()
    produce_heatmap(model, True, True)
    process(model, totals, True, True)
