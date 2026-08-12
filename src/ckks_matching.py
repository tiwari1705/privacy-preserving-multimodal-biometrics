"""
CKKS encrypted similarity utilities using TenSEAL.

This module provides the implementation interface.
TenSEAL must be installed separately.
"""

import numpy as np


def create_ckks_context():
    """
    Create a CKKS context.

    Parameters follow the final project configuration.
    """
    import tenseal as ts

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )

    context.global_scale = 2 ** 40

    return context


def encrypt_vector(context, vector):
    """Encrypt a real-valued vector."""
    import tenseal as ts

    vector = np.asarray(
        vector,
        dtype=np.float64
    ).tolist()

    return ts.ckks_vector(
        context,
        vector
    )


def encrypted_inner_product(
    encrypted_vector,
    plaintext_vector
):
    """
    Compute an encrypted/plaintext inner product.

    The result remains encrypted until decryption.
    """
    plaintext_vector = np.asarray(
        plaintext_vector,
        dtype=np.float64
    ).tolist()

    encrypted_product = (
        encrypted_vector
        * plaintext_vector
    )

    return encrypted_product.sum()


def decrypt_score(
    encrypted_score,
    secret_context
):
    """Decrypt the encrypted similarity score."""
    return float(
        encrypted_score.decrypt(
            secret_context.secret_key()
        )[0]
    )
