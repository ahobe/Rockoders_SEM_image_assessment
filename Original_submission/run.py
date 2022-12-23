from assesSEM.pipelines import run_original_pipeline

# Ensure that this is called from the base directory of the dataset folders.

model_name = "model_mlo_512_512_2.h5"
model_name = "model_mlo_512_512_unshifted.h5"
model_name = "model_mlo_512_512_unshifted_mm.h5"

run_original_pipeline(model_name)

