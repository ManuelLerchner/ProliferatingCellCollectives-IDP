import random
import numpy as np
from typing import Dict, Union, Optional, Generator


def physics_config_generator(
    base_config: Dict[str, float],
    variation: Union[float, Dict[str,
                                 Union[float, Dict[str, Union[float, str]]]]] = 0.1,
    num_samples: Optional[int] = None,
    distribution: str = 'uniform',
    seed: Optional[int] = None
) -> Generator[Dict[str, float], None, None]:
    """
    Generate variations of physics configuration parameters.

    Args:
        base_config: Base configuration dictionary with parameter names and values
        variation: 
            - float: uniform variation percentage for all parameters
            - dict: per-parameter variation specification:
                {"param": 0.2} → ±20% variation
                {"param": {"type": "percent", "value": 0.2}} → explicit percentage
                {"param": {"type": "magnitude", "value": 1}} → vary ±1 order of magnitude
        num_samples: Number of samples to generate. If None, generates indefinitely
        distribution: 'uniform' or 'normal' - type of random distribution
        seed: Random seed for reproducibility

    Yields:
        Dict[str, float]: Configuration with varied parameters
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Normalize variation spec into dict with type + value
    variation_spec = {}
    if isinstance(variation, (int, float)):
        for key in base_config.keys():
            variation_spec[key] = {
                "type": "percent", "value": float(variation)}
    else:
        for key, v in variation.items():
            if isinstance(v, (int, float)):
                variation_spec[key] = {"type": "percent", "value": float(v)}
            elif isinstance(v, dict):
                if "type" not in v or "value" not in v:
                    raise ValueError(f"Invalid variation spec for {key}: {v}")
                variation_spec[key] = v
            else:
                raise ValueError(f"Unsupported variation value for {key}: {v}")

    count = 0
    while num_samples is None or count < num_samples:
        varied_config = {}

        for param_name, base_value in base_config.items():
            spec = variation_spec.get(
                param_name, {"type": "percent", "value": 0.1})

            if spec["type"] == "percent":
                variation_pct = spec["value"]

                if distribution == "uniform":
                    min_val = base_value * (1 - variation_pct)
                    max_val = base_value * (1 + variation_pct)
                    varied_value = random.uniform(min_val, max_val)

                elif distribution == "normal":
                    std_dev = base_value * variation_pct / 3
                    varied_value = np.random.normal(base_value, std_dev)
                    varied_value = max(varied_value, base_value * 0.01)

                else:
                    raise ValueError(f"Unknown distribution: {distribution}")

            elif spec["type"] == "magnitude":
                mag_range = spec["value"]
                log_base = np.log10(base_value)

                if distribution == "uniform":
                    varied_log = random.uniform(
                        log_base - mag_range, log_base + mag_range)
                elif distribution == "normal":
                    std_dev = mag_range / 3
                    varied_log = np.random.normal(log_base, std_dev)
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")

                varied_value = 10 ** varied_log

            else:
                raise ValueError(f"Unknown variation type: {spec['type']}")

            varied_config[param_name] = varied_value

        yield varied_config
        count += 1
