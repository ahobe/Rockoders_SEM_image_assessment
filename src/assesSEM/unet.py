# Building Unet by dividing encoder and decoder into blocks
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

from assesSEM.IO import read_and_normalize_image


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)  # Not in the original network.
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  # Not in the original network
    x = Activation("relu")(x)

    return x


# Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


# Decoder block
# skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Build Unet using the blocks
def build_unet(input_shape, n_classes, name):
    if name == 'default':
        name = "model_mlo_512_512_2.h5"
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)  # Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  # Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(
        d4)  # Change the activation based on n_classes
    # print(activation)

    model = Model(inputs, outputs, name=name)
    return model


class ModelAttributes:
    def __init__(self, p_h=512, p_w=512, n_channels=2, nb_classes=5):
        self.patch_height = p_h
        self.patch_width = p_w
        self.no_of_channels = n_channels
        self.no_of_classes = nb_classes


def get_model_shape_and_classes(name='default'):
    if name == 'default' or name == "model_mlo_512_512_2.h5" or name == "model_mlo_512_512_unshifted.h5":
        model_params = ModelAttributes()
    elif name == "model_mlo_512_512_unshifted_mm.h5":
        model_params = ModelAttributes(n_channels=3)
    else:
        return ValueError

    input_shape = (model_params.patch_height, model_params.patch_width, model_params.no_of_channels)

    return model_params.no_of_classes, input_shape


def create_unet_input(bse_im, cl_im, mm_im=None):
    if mm_im is not None:
        x = np.zeros([cl_im.shape[0], cl_im.shape[1], 3])
    else:
        x = np.zeros([cl_im.shape[0], cl_im.shape[1], 2])
    x[:, :, 0] = cl_im
    x[:, :, 1] = bse_im
    if mm_im is not None:
        x[:, :, 2] = mm_im
    return x


def get_unet_input(model_name, image_metadata):
    if model_name == "model_mlo_512_512_unshifted_mm.h5" :
        image_path_mm = image_metadata.image_path_mm
    elif model_name == 'default' or model_name == "model_mlo_512_512_2.h5" or \
            model_name == "model_mlo_512_512_unshifted.h5":
        image_path_mm = None
    else:
        raise ValueError(str(model_name))
    cl_im = read_and_normalize_image(image_metadata.image_path_cl)
    bse_im = read_and_normalize_image(image_metadata.image_path_bse)

    if image_path_mm:  # not None
        mm_im = read_and_normalize_image(image_path_mm)
        unet_input = create_unet_input(bse_im, cl_im, mm_im=mm_im)
    else:
        unet_input = create_unet_input(bse_im, cl_im)

    return unet_input
