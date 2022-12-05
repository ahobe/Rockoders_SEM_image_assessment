from importlib.resources import files

from assesSEM.unet import get_model_shape_and_classes, build_unet


def build_and_load_existing_model(name="model_mlo_512_512_2.h5"):
    if name == "default" or name == "model_mlo_512_512_2.h5":
        nb_classes, input_shape, image_height = get_model_shape_and_classes()
        current_model = build_unet(input_shape, n_classes=nb_classes)
        current_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_path = files('assesSEM.models').joinpath("model_mlo_512_512_2.h5")
        current_model.load_weights(model_path)
    else:
        raise ValueError
    return current_model, nb_classes, image_height