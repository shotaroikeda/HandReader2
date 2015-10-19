from __future__ import unicode_literals, print_function, division, absolute_import

# imports required for the script
import numpy as np
import struct


# Constant path variables
PATH_TO_MNIST = "../mnist/"
PATH_TO_TRAINING_IMGS = PATH_TO_MNIST + "train-images.idx3-ubyte"
PATH_TO_TRAINING_LABELS = PATH_TO_MNIST + "train-labels.idx1-ubyte"
PATH_TO_T10K_IMGS = PATH_TO_MNIST + "t10k-images.idx3-ubyte"
PATH_TO_T10K_LABELS = PATH_TO_MNIST + "t10k-labels.idx1-ubyte"


# Custom Error
class MnistBadArgumentError(Exception):
    pass


def read_num(file_to_process=None, v=False):
    """
    Read_num will return a tuple of an array and the number associated with it.

    @args:
    Takes 1 argument. either 'training' or 'test'

    ex.
    (np.array([[0, 0, 0, ...0], ...[....,0]]), 3)

    This means that the number that the array represents is a 3.

    Note:
    This uses a generator.
    To process, you will have to use either a for loop, or use .next().
    """

    if file_to_process is None:
        raise MnistBadArgumentError(
            'Function read_num() takes 1 argument. Please enter either \'training\' or \'test\'.'
        )

    f_imgs = None
    f_labels = None

    if file_to_process == 'training':
        f_imgs = open(PATH_TO_TRAINING_IMGS, 'rb')
        f_labels = open(PATH_TO_TRAINING_LABELS, 'rb')

    elif file_to_process == 'test':
        f_imgs = open(PATH_TO_T10K_IMGS, 'rb')
        f_labels = open(PATH_TO_T10K_LABELS, 'rb')

    else:
        raise MnistBadArgumentError(
            'There is no such file to process. Enter either \'training\' or \'test\' (case sensitve)'
        )

    magic_num_imgs = struct.unpack('>i', f_imgs.read(4))
    magic_num_labels = struct.unpack('>i', f_labels.read(4))

    if v:
        print("Reading Image file with Magic number: %d" % (magic_num_imgs))
        print("Reading Label file with Magic number: %d" % (magic_num_labels))

    total_set_imgs = struct.unpack('>i', f_imgs.read(4))[0]
    total_set_labels = struct.unpack('>i', f_labels.read(4))[0]

    if v:
        print("Total amount of images in the set: %d" % (total_set_imgs))
        print("Total amount of labels in the set: %d" % (total_set_labels))

    if total_set_imgs != total_set_labels:
        raise ValueError('The set numbers do not match!')

    num_row = struct.unpack('>i', f_imgs.read(4))[0]
    num_col = struct.unpack('>i', f_imgs.read(4))[0]

    if v:
        print("Number of rows for the image: %dpxs" % (num_row))
        print("Number of rows for the image: %dpxs" % (num_col))

    for i in range(total_set_imgs):
        arr = np.array([
            [1 if struct.unpack('>?', f_imgs.read(1))[0] else 0
            for i in range(num_row)]
            for i in range(num_col)
            ])

        num = struct.unpack('>B', f_labels.read(1))[0]

        yield (arr, num)
