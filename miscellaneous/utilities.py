from IPPy import utilities as IPutils


def create_simulation(K, x_true, noise_level, device):
    # --- Load data ---
    img_size_to_use = config.get("image_size", model.config.sample_size)
    if isinstance(img_size_to_use, int):
        img_size_to_use = (img_size_to_use, img_size_to_use)
    logger.info(f"Target image size set to: {img_size_to_use}")

    x_true = utilities.load_and_preprocess_image(
        config, img_size_to_use, device, logger
    )
    img_shape_hw = x_true.shape[-2:]
    num_channels = config["image_channels"]

    # --- Define operator & Generate test problem ---
    K = model_setup.get_operator(config, img_shape_hw, num_channels, device, logger)

    logger.info("Generating test problem (corrupted data y_delta)...")
    y = K(x_true)
    y_delta = y + IPutils.gaussian_noise(y, config["noise_level"])
    logger.info(f"Test problem generated. y_delta shape: {y_delta.shape}")

    return K, x_true, y_delta
