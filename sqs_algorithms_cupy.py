'''
This file provide general utilities in sqs algorithm using generic forward and backprojector.

Change all the cupy to numpy should be able to change it to numpy version.
'''

from typing import Union, Tuple, Callable, Any

import cupy as cp
import prior.recon_prior_cupy as recon_prior
from projector.ct_projector_cupy import ct_projector


def sqs_gaussian_one_step(
    projector: ct_projector,
    img: cp.array,
    prj: cp.array,
    norm_img: cp.array,
    projector_norm: float,
    beta: float,
    weight: cp.array = None,
    return_loss: bool = False
) -> Union[cp.array, Tuple[cp.array, float, float]]:
    '''
    sqs with gaussian prior. Please see the doc/sqs_equations, section 4.

    Parameters
    -----------------
    projector: ct_projector.
        The projector to perform forward and back projections. Its parameter should be
        compatible with the shape of img and prj.
    img: array(float32) of shape [batch, nz, ny, nx].
        The current image.
    prj: array(float32) of shape [batch, nview, nv, nu].
        The projection data.
    norm_img: array(float32) of shape [batch, nz, ny, nx].
        The norm image A^T*w*A*1.
    projector_norm: float.
        The norm of the system matrix, to make penalty parameter easier to tune.
    beta: float.
        Gaussian prior strength.
    weight: array(float32) of shape [batch, nview, nv, nu].
        The weighting matrix.
    return_loss: bool
        If true, the function will return a tuple [result image, data_loss, penalty_loss].
        If false, only the result image will be returned.

    Returns
    -------------------
    img: array(float32) of shape [batch, nz, ny, nx].
        The updated image.
    data_loss: float.
        Only return if return_loss is True. The data term loss.
    nlm_loss: float.
        Only return if return_loss is True. The penalty term loss.
    '''
    def gaussian_func(img):
        return recon_prior.nlm(img, cp.ones(img.shape, cp.float32), 1, [3, 3, 3], [1, 1, 1], 1)

    if weight is None:
        weight = 1

    # A.Tw(Ax)
    fp = projector.fp(img) / projector_norm
    fp = fp - prj / projector_norm
    bp = projector.bp(fp * weight) / projector_norm

    # sqs
    gauss = 4 * (img - gaussian_func(img))
    img = img - (bp + beta * gauss) / (norm_img + beta * 8)

    if return_loss:
        fp = projector.fp(img) / projector_norm
        data_loss = 0.5 * cp.sum(weight * (fp - prj / projector_norm)**2)

        nlm = gaussian_func(img)
        nlm2 = gaussian_func(img * img)
        nlm_loss = cp.sum(img * img - 2 * img * nlm + nlm2)

        return img, data_loss, nlm_loss
    else:
        return img


def nesterov_acceleration(
    func: Callable[..., Union[cp.array, Tuple[Any, ...]]],
    img: cp.array,
    img_nesterov: cp.array,
    nesterov: float = 0.5,
    **kwargs
) -> Tuple[cp.array, cp.array]:
    '''
    kwargs should contains all params for func except for img.
    func should return img as the only or the first return value.

    Parameters
    -------------------
    func: callable
        The iteration function. It must take a kwarg 'img' as the current image.
        The return value must be a single img array, or a tuple with the first
        element as img.
    img: np.array(float32)
        The current image, whose shape should be compatible with func.
    img_nesterov: np.array(float32)
        The nesterov image (x*), whose shape should be the same with img.
    nesterov: float
        The acceleration parameter.
    **kwargs: dict
        The keyword argument list to be passed to func. Note that 'img' will
        be overridden by img_nesterov.

    Returns
    --------------------
    img: np.array(float32).
        The updated image.
    img_nesterov: np.array(float32).
        The nesterov image (x*) for the next iteration.
    '''

    kwargs['img'] = img_nesterov
    res = func(**kwargs)

    if type(res) is tuple:
        img_nesterov = res[0] + nesterov * (res[0] - img_nesterov)
        img = res[0]
    else:
        img_nesterov = res + nesterov * (res - img_nesterov)
        img = res

    return img, img_nesterov
