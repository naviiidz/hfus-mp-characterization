def clean_radius(radius):
    """Clean radius values from strings to floats."""
    if isinstance(radius, str):
        return float(radius.strip('[]').strip())
    return float(radius)

def get_bin_label(size, bin_ranges):
    """Assign a bin label based on size ranges."""
    for i, (min_size, max_size) in enumerate(bin_ranges):
        if min_size <= size <= max_size:
            return i
    return None