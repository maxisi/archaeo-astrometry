#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import yaml
import logging


def dN(h, n, q):
    """Displacement induced by gravitational wave, Eq. (20) in
    arXiv:1804.00660

    Arguments
    ---------
    h : array_like
        gravitational wave strain tensor (3x3)
    n : array_like
        direction to the source.
    q : array_like
        wave propagation direction.
    """
    n = np.array(n)
    q = np.array(q)
    h = np.array(h)

    # term1 = (n - q) / (1 - np.dot(q, n))
    norm = (1 - np.einsum('...i,...i', q, n))
    if n.ndim > 1:
        norm = norm[..., None]
    term1 = (n - q) / norm

    term2 = np.einsum('...jk,...j,...k', h, n, n)

    term3 = np.einsum('...ij,...j', h, n)

    missing_dims = term1.ndim - np.ndim(term2)
    for i in range(missing_dims):
        term2 = term2[..., None]

    result = 0.5 * (term1 * term2 - term3)

    return result


# GW polarization tensors
# WARNING: these expressions assume GW propagates in the z direction, so we
# must have q = (0, 0, 1)
ep = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
ec = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])


# GW waveform
def gw(ap, ac, phi):
    """Gravitational wave strain tensor (circularly polarized wave).

    Arguments
    ---------
    ap : float
        amplitude of the plus polarization
    ac : float
        amplitude of the cross polarization
    phi : float
        phase of the wave
    """
    # expand dimensions to broadcast GW tensor
    phi = np.array(phi)
    ap_cos = ap * np.cos(phi)
    ac_sin = ac * np.sin(phi)
    
    _ep = ep
    _ec = ec
    for d in phi.shape:
        _ep = _ep[None, ...]
        _ec = _ec[None, ...]
    if phi.ndim > 0:
        ap_cos = ap_cos[..., None, None]
        ac_sin = ac_sin[..., None, None]

    return _ep * ap_cos + _ec * ac_sin


def dstar(n, phi=0, ap=0.1, ac=0):
    """ Displacement of a star due to a gravitational wave
    propagating in the z direction.
    """
    n = np.atleast_2d(n)  # (n_N, 3)
    phi = np.atleast_1d(phi)  # (n_phi,)

    if isinstance(ap, (int, float)):
        ap = ap*np.ones_like(phi)
    elif len(ap) == len(phi):
        ap = np.atleast_1d(ap)  # (n_phi,)
    else:
        raise ValueError("ap must be a scalar or have the same length as phi")

    if isinstance(ac, (int, float)):
        ac = ac*np.ones_like(phi)
    elif len(ac) == len(phi):
        ac = np.atleast_1d(ac)
    else:
        raise ValueError("ac must be a scalar or have the same length as phi")
    
    if n.ndim > 2 or n.shape[-1] != 3:
        raise ValueError("n must have shape (3,) or (n, 3)")

    if phi.ndim > 1:
        raise ValueError("phi must have shape (n,) or ()")

    n = n[None, ...]  # (1, n_N, 3)
    phi = phi[..., None]  # (n_phi, 1)
    ap = ap[..., None]  # (n_phi, 1)
    ac = ac[..., None]  # (n_phi, 1)
    # gw will add two more dimensions so that phi ends up
    # (n_phi, 1, 1, 1)
    # so that it can be broadasted to the 3x3 GW tensors

    q = np.array([0, 0, 1])*np.ones_like(n)

    return np.squeeze(dN(gw(ap, ac, phi), n, q))


if __name__ == "__main__":

    # load options from YAML
    with open('config_chirp.yml') as f:
        config = yaml.safe_load(f)

    if 'input' in config:
        input_path = config['input']
        if os.path.exists(input_path):
            stars = np.loadtxt(input_path, delimiter=config.get('sep', None))
        else:
            raise FileNotFoundError(f"File {input_path} not found.")
    elif 'star_number' in config:
        nstars = int(config['star_number'])
        rng = np.random.default_rng(int(config.get('seed', 150914)))
        stars = np.random.randn(nstars, 3)
        stars /= np.linalg.norm(stars, axis=-1)[..., None]
    else:
        raise ValueError("No input file specified in config.yml")

    if 'time_samples' in config:
        nsamp = int(config['time_samples'])
    else:
        nsamp = 100
        logging.warning("No 'time_samples' in config.yml. Defaulting"
                        f"to {nsamp}.")

    if 'duration' in config:
        duration = float(config['duration'])
    else:
        duration = 1

    if 'chirp_mass' in config:
        chirp_mass = float(config['chirp_mass'])
    else:
        chirp_mass = 10.
        logging.warning("No 'chirp_mass' in config.yml."
                        f"Defaulting to {chirp_mass}.")

    c = 299792458  # speed of light
    G = 6.67430e-11  # Newton's constant
    msun = 1.989e30  # solar mass
    chirp_mass *= msun
    chirp_radius = G * chirp_mass / c**2

    tau = np.linspace(duration, 0, nsamp, endpoint=False)
    f = 2*(5/256/tau)**(3/8) / (chirp_radius/c)**(5/8)
    phis = 2*np.pi*f*tau

    if 'distance_mpc' in config:
        dist_mpc = float(config['distance_mpc'])
    else:
        dist_mpc = 1.
        logging.warning("No 'distance_mpc' in config.yml."
                        f"Defaulting to {dist_mpc} Mpc.")

    if 'amplitude_scale' in config:
        amp_scale = float(config['amplitude_scale'])
    else:
        amp_scale = 1.
        logging.warning("No 'amplitude_scale' in config.yml."
                        f"Defaulting to {amp_scale}.")
        
    distance = dist_mpc * 3.08567758149137e22  # Mpc to meters
    amp = chirp_radius * (5*chirp_radius/c/tau)**(1/4) / distance
    amp *= amp_scale

    # get displacements for all our stars
    # this will be a (nsamp, nstars, 3) array
    d = dstar(stars, phis, amp, 0)
    print(f"Maximum absolute deviation: {np.max(np.abs(d))}")

    # update the star locations
    # this will still be a (nsamp, nstars, 3) array
    newstars = stars + d
    newstars /= np.linalg.norm(newstars, axis=-1)[..., None]

    # stack array to get (nstars, 3*nsamp)
    newstars = np.hstack(newstars)
    h = ','.join([f"X_frame{i},Y_frame{i},Z_frame{i}"
                  for i in range(newstars.shape[1]//3)])
    np.savetxt('output_chirp.csv', newstars, fmt='%.15f',
               delimiter=',', header=h)
