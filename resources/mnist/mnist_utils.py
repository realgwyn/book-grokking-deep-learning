import numpy
import os
import functools
import operator
import gzip
import struct
import array


def load_test_images():
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'test-images-idx3-ubyte.gz')
    print(f"Loading test images from: {path}")
    return parse_idx(path)

def load_test_labels():
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'test-labels-idx1-ubyte.gz')
    print(f"Loading test labels from: {path}")
    return parse_idx(path)

def load_training_images():
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'training-images-idx3-ubyte.gz')
    print(f"Loading training images from: {path}")
    return parse_idx(path)

def load_training_labels():
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'training-labels-idx1-ubyte.gz')
    print(f"Loading training labels from: {path}")
    return parse_idx(path)

def parse_idx(fname):
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                      0x09: 'b',  # signed byte
                      0x0b: 'h',  # short (2 bytes)
                      0x0c: 'i',  # int (4 bytes)
                      0x0d: 'f',  # float (4 bytes)
                      0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        if len(header) != 4:
            raise RuntimeError('Invalid IDX file, '
                                 'file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise RuntimeError('Invalid IDX file, '
                                 'file must start with two zero bytes. '
                                 'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise RuntimeError('Unknown data type '
                                 '0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise RuntimeError('IDX file has wrong number of items. '
                                 'Expected: %d. Found: %d' % (expected_items,
                                                              len(data)))

        output_array = numpy.array(data).reshape(dimension_sizes)
        print(f"Dataset Shape: {output_array.shape}")



        return output_array

def print_array(image):
    print(image.reshape(28, 28))

def print_bitmap(image):
    for row in image:
        print("".join(bitmap_value_to_symbol(val) for val in row))

def bitmap_value_to_symbol(val):
    if  val == 0:
        return "  "
    elif val < 100:
        return "░░"
    elif val < 150:
        return "▒▒"
    elif val < 200:
        return "▓▓"
    else:
        return "██"

def print_float_image(image):
    for row in image:
        print("".join(float_value_to_symbol(val) for val in row))

def float_value_to_symbol(val):
    if  val == 0:
        return "  "
    elif val < 0.25:
        return "░░"
    elif val < 0.50:
        return "▒▒"
    elif val < 0.75:
        return "▓▓"
    else:
        return "██"

def print_weights_image(image):
    for row in image:
        print("".join(weight_to_symbol(val) for val in row))

def weight_to_symbol(val):
    if  val == 0:
        return "  "
    elif val < 0.0001:
        return "░░"
    elif val < 0.001:
        return "▒▒"
    elif val < 0.01:
        return "▓▓"
    else:
        return "██"