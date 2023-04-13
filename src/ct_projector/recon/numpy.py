import ct_projector.recon as recon

recon.BACKEND = 'numpy'

from .sqs_algorithms import sqs_gaussian_one_step, nesterov_acceleration, sqs_one_step  # noqa
from . import get_backend  # noqa
