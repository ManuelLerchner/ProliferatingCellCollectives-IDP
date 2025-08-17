import random
from typing import Dict, Generator, Optional, Union

import numpy as np


def physics_config_generator(
    base_config: Dict[str, float],
    variation_percent: Union[float, Dict[str, float]] = 0.1,
    num_samples: Optional[int] = None,
    distribution: str = 'uniform',
    seed: Optional[int] = None
) -> Generator[Dict[str, float], None, None]:
    """
    Generate variations of physics configuration parameters.

    Args:
        base_config: Base configuration dictionary with parameter names and values
        variation_percent: Either a single float for uniform variation across all params,
                          or a dict mapping param names to their specific variation percentages
        num_samples: Number of samples to generate. If None, generates indefinitely
        distribution: 'uniform' or 'normal' - type of random distribution
        seed: Random seed for reproducibility

    Yields:
        Dict[str, float]: Configuration with varied parameters
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Convert single variation_percent to dict
    if isinstance(variation_percent, (int, float)):
        variation_dict = {key: variation_percent for key in base_config.keys()}
    else:
        variation_dict = variation_percent

    count = 0
    while num_samples is None or count < num_samples:
        varied_config = {}

        for param_name, base_value in base_config.items():
            # Default to 10% if not specified
            variation_pct = variation_dict.get(param_name, 0.1)

            if distribution == 'uniform':
                # Uniform distribution: ±variation_percent around base value
                min_val = base_value * (1 - variation_pct)
                max_val = base_value * (1 + variation_pct)
                varied_value = random.uniform(min_val, max_val)

            elif distribution == 'normal':
                # Normal distribution: mean=base_value, std=base_value*variation_percent/3
                # This gives ~99.7% of values within ±variation_percent
                std_dev = base_value * variation_pct / 3
                varied_value = np.random.normal(base_value, std_dev)

                # Ensure positive values for physical parameters
                # Minimum 1% of base value
                varied_value = max(varied_value, base_value * 0.01)

            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            varied_config[param_name] = varied_value

        yield varied_config
        count += 1
