def isinstance_qutip_qobj(obj):
    """Check if the object is a qutip Qobj.

    Args:
        obj (any): Any object for testing.

    Returns:
        Bool: True if obj is qutip Qobj
    """
    if (
        type(obj).__name__ == "Qobj"
        and hasattr(obj, "_data")
        and type(obj._data).__name__ == "fast_csr_matrix"
    ):
        return True
    return False
