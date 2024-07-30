import numpy as np

def quantize_lloyd_max(block, num_levels=32, num_iterations=10):
    block_size = block.shape[0]
    dc_coefficient = block[0, 0]
    ac_coefficients = block.copy()
    ac_coefficients[0, 0] = 0
    max_ac_coefficient = np.max(np.abs(ac_coefficients))
    lower_limit = np.min(ac_coefficients)
    levels = np.linspace(lower_limit, max_ac_coefficient, num_levels)

    for _ in range(num_iterations):
        ac_coefficients_flat = ac_coefficients.flatten()
        indices = np.argmin(np.abs(ac_coefficients_flat[:, np.newaxis] - levels), axis=1)
        quantized_block_flat = levels[indices]
        for k in range(num_levels):
            level_samples = ac_coefficients_flat[indices == k]
            if len(level_samples) > 0:
                mu = np.mean(level_samples)
                sigma = np.std(level_samples)
                levels[k] = mu + sigma * np.random.randn()
        quantized_block = quantized_block_flat.reshape(block.shape)
    quantized_block[0, 0] = dc_coefficient

    return quantized_block