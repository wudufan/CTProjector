# This file provide general utilities in sqs algorithm using forward and backprojector

import cupy as cp
import prior.recon_prior_cupy as recon_prior

def sqs_gaussian_one_step(projector, img, prj, norm_img, projector_norm, beta, weight = None, return_loss = False):
    '''
    sqs with gaussian prior. Please see the doc/sqs_equations, section 4
    '''
    def gaussian_func(img):
        return recon_prior.nlm(img, cp.ones(img.shape, cp.float32), 1, [3,3,3], [1,1,1], 1)

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



