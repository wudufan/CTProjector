'''
The old parallel conebeam rebinning for comparison
'''

# %%
import numpy as np
import h5py
import sys

# %%
import argparse
parser = argparse.ArgumentParser(description='rebin helical conebeam to parallel')
parser.add_argument('--src', dest='src', default=None)
parser.add_argument('--out', dest='out', default='rebin')
parser.add_argument('--set', dest='set', default='all')

# %%
if 'ipykernel' in sys.argv[0]:
    args = parser.parse_args([
        '--src', '/home/local/PARTNERS/dw640/CTProjector/example/54_1.mat',
        '--set', 'all'
    ])
else:
    args = parser.parse_args()

for k in vars(args):
    print(k, '=', getattr(args, k), flush=True)


# %%
def ConvertAnglesAndCalulcatePitch(angles, poses):
    for i in range(1, len(angles)):
        if angles[i] < angles[i - 1]:
            angles[i:] += 2 * np.pi

    coef = np.polyfit(angles, poses, 1)
    zrot = coef[0] * 2 * np.pi

    return angles, zrot


# %%
def GetRebinParameters(nu, off_u, dso, da, dTheta):
    db = da * dso

    cu = (nu - 1) / 2 + off_u
    beta_min = -cu * da
    beta_max = (nu - 1 - cu) * da
    bmin = dso * np.sin(beta_min)
    bmax = dso * np.sin(beta_max)
    # shrink the fov a little so that there will be no gap when padding B with A
    cb = np.floor(-bmin / db)
    nb = int(np.floor(-bmin / db) + np.floor(bmax / db))
    off_b = cb - (nb - 1) / 2

    # get theta range
    nRequireViewPre = int(np.ceil(beta_max / dTheta))
    nRequireViewPost = int(np.ceil(-beta_min / dTheta))

    return nb, off_b, nRequireViewPre, nRequireViewPost


# %%
def RebinToParallelConebeam(
    prj, cu, nb, cb, db, dso, dAngle, nTheta, theta0, tubeAngles, iStart, iEnd
):
    bs = (np.arange(nb) - cb) * db
    betas = np.arcsin(bs / dso)
    thetas = np.arange(nTheta) * dAngle + theta0

    # interpolate betas first
    print('Beta interpolation...')
    us = betas / da + cu
    rprjBeta = np.zeros([nb, prj.shape[1], prj.shape[2]], np.float32)
    for ib in range(nb):
        if (ib + 1) % 100 == 0:
            print(ib + 1, end=',', flush=True)

        u = us[ib]
        u0 = int(u)
        u1 = u0 + 1
        w = u - u0

        if u0 >= 0 and u0 < nua:
            rprjBeta[ib, ...] += (1 - w) * prj[u0, ...]
        if u1 >= 0 and u1 < nua:
            rprjBeta[ib, ...] += w * prj[u1, ...]
    print('')

    # then interpolate theta
    print('Theta interpolation...')
    rprj = np.zeros([nb, len(thetas), nv], np.float32)
    for ib in range(nb):
        if (ib + 1) % 100 == 0:
            print(ib + 1, end=',')

        alphas = (thetas - betas[ib] - tubeAngles[iStart]) / dAngle

        alphas0 = alphas.astype(int)
        alphas1 = alphas0 + 1
        w = alphas - alphas0

        validInds0 = np.where((alphas0 > 0) & (alphas0 < iEnd - iStart))[0]
        alphas0 = alphas0[validInds0]
        rprj[ib, validInds0, :] += (1 - w[validInds0][:, np.newaxis]) * rprjBeta[ib, alphas0, :]

        validInds1 = np.where((alphas1 > 0) & (alphas1 < iEnd - iStart))[0]
        alphas1 = alphas1[validInds1]
        rprj[ib, validInds1, :] += w[validInds1][:, np.newaxis] * rprjBeta[ib, alphas1, :]
    print('')

    return rprj


# %%
dataPath = args.src
with h5py.File(dataPath, 'r') as f:
    lookupA = np.copy(f['sh']['Lookup']['DetA'])
    lookupB = np.copy(f['sh']['Lookup']['DetB'])

    # convert to mm
    posA = np.copy(f['posA']).flatten() / 1000
    posB = np.copy(f['posB']).flatten() / 1000

    # convert to rad
    anglesA = np.copy(f['angleA']).flatten() / 180 * np.pi
    anglesB = np.copy(f['angleB']).flatten() / 180 * np.pi

    # convert to attenuation
    prjA = np.copy(np.transpose(f['projA'], (2, 0, 1))[..., ::-1], 'C') / 2294.5
    prjB = np.copy(np.transpose(f['projB'], (2, 0, 1))[..., ::-1], 'C') / 2294.5

# %%
prjSet = args.set

if prjSet == 'even':
    lookupA = lookupA[::2]
    lookupB = lookupB[::2]
    posA = posA[::2]
    posB = posB[::2]
    anglesA = anglesA[::2]
    anglesB = anglesB[::2]
    prjA = prjA[:, ::2, :]
    prjB = prjB[:, ::2, :]
elif prjSet == 'odd':
    lookupA = lookupA[1::2]
    lookupB = lookupB[1::2]
    posA = posA[1::2]
    posB = posB[1::2]
    anglesA = anglesA[1::2]
    anglesB = anglesB[1::2]
    prjA = prjA[:, 1::2, :]
    prjB = prjB[:, 1::2, :]

# %%
# parameters
nua = prjA.shape[0]
nub = prjB.shape[0]
nv = prjA.shape[-1]
dso = 595
da = 0.067864004196156 * np.pi / 180
dv = 1.0947
off_u = -1.25
if prjSet == 'even' or prjSet == 'odd':
    rotview = int(1152 / 2)
else:
    rotview = 1152
dTheta = np.pi * 2 / rotview

anglesA, zrotA = ConvertAnglesAndCalulcatePitch(anglesA, posA)
anglesB, zrotB = ConvertAnglesAndCalulcatePitch(anglesB, posB)

zrot = (zrotA + zrotB) / 2

# %%
# get rebinning parameters
nba, off_ba, nrviewPre_a, nrviewPost_a = GetRebinParameters(nua, off_u, dso, da, dTheta)
nbb, off_bb, nrviewPre_b, nrviewPost_b = GetRebinParameters(nub, off_u, dso, da, dTheta)
cua = (nua - 1) / 2 + off_u
cub = (nub - 1) / 2 + off_u
cba = (nba - 1) / 2 + off_ba
cbb = (nbb - 1) / 2 + off_bb
db = da * dso

# %%
# get padding parameters
iStartA = np.where(lookupA > 0.5)[0][0]
iEndA = np.where(lookupA > 0.5)[0][-1] + 1

iStartB = np.where(lookupB > 0.5)[0][0]
iEndB = np.where(lookupB > 0.5)[0][-1] + 1

assert(iStartB >= iStartA and iEndB <= iEndA)

# calculate offset on A to pad B, from both same direction and opposite direction
angleBMinusA = anglesB[0] - anglesA[iStartB - iStartA]
angleBMinusA = angleBMinusA - int(angleBMinusA / np.pi) * np.pi
iPrjOffsetASameDir = int(angleBMinusA / dTheta)

zOffsetASameDir = angleBMinusA * zrot / 2 / np.pi

if iPrjOffsetASameDir > 0:
    iPrjOffsetAOppDir = iPrjOffsetASameDir - int(rotview / 2)
    zOffsetAOppDir = zOffsetASameDir - zrot / 2
else:
    iPrjOffsetAOppDir = iPrjOffsetASameDir + int(rotview / 2)
    zOffsetAOppDir = zOffsetASameDir + zrot / 2

# calculate the start and end of parallel rebinned A to pad B
iPrjOffsetAStart = min(iPrjOffsetASameDir, iPrjOffsetAOppDir)
iPrjOffsetAEnd = max(iPrjOffsetASameDir, iPrjOffsetAOppDir)

# %%
# calculate absolute positions
iPrjAStart = iStartA
iPrjBStart = iStartB
iParaAStart = iPrjAStart + nrviewPre_a
iParaBStart = iPrjBStart + nrviewPre_b
iPadAStart = iParaBStart + iPrjOffsetAStart

iPrjAEnd = iEndA
iPrjBEnd = iEndB
iParaAEnd = iPrjAEnd - nrviewPost_a
iParaBEnd = iPrjBEnd - nrviewPost_b
iPadAEnd = iParaBEnd + iPrjOffsetAEnd

# rebin relative positions
iRebinAStart = iParaAStart - iPrjAStart
iRebinAEnd = iParaAEnd - iPrjAStart
iRebinBStart = iParaBStart - iPrjBStart
iRebinBEnd = iParaBEnd - iPrjBStart

# padding, give positions relatively in the parallel beam coordinate
iParaAStartRel = iPadAStart - iParaAStart
iParaAEndRel = iPadAEnd - iParaAStart
if iParaAStartRel < 0:
    iParaBStartRel = -iParaAStartRel
    iParaAStartRel = 0
else:
    iParaBStartRel = 0
if iParaAEndRel > (iParaAEnd - iParaAStart):
    iParaAEndPad = 0
    iParaBEndRel = (iParaBEnd - iParaBStart) - (iParaAEndPad - (iParaAEnd - iParaAStart))
    iParaAEndRel = iParaAEnd - iParaAStart
else:
    iParaBEndRel = iParaBEnd - iParaBStart

iParaAStartFinalRel = iParaBStartRel + iParaBStart - iParaAStart

# %%
paraA = RebinToParallelConebeam(
    prjA, cua, nba, cba, db, dso, dTheta, iRebinAEnd - iRebinAStart,
    anglesA[iRebinAStart], anglesA, 0, iPrjAEnd - iPrjAStart
)

# %%
paraB = RebinToParallelConebeam(
    prjB, cub, nbb, cbb, db, dso, dTheta, iRebinBEnd - iRebinBStart,
    anglesB[iRebinBStart], anglesB, 0, iPrjBEnd - iPrjBStart
)

# %%
# transit length
L = 20
w = np.cos(np.pi / 2 * np.arange(L) / L)
w = w * w
w = w[..., np.newaxis, np.newaxis]

# %%
# pad paraB with paraA
paraBEx = np.zeros([paraA.shape[0], iParaBEndRel - iParaBStartRel, paraB.shape[2]], np.float32)

# padding different direction
if zOffsetAOppDir < 0:
    iv = int(-zOffsetAOppDir / dv)
    paraBEx[..., :-iv] = paraA[
        ::-1,
        (iParaAStartFinalRel + iPrjOffsetAOppDir):(iParaAStartFinalRel + paraBEx.shape[1] + iPrjOffsetAOppDir),
        iv:
    ]
else:
    iv = int(zOffsetAOppDir / dv)
    paraBEx[..., iv:] = paraA[
        ::-1,
        (iParaAStartFinalRel + iPrjOffsetAOppDir):(iParaAStartFinalRel + paraBEx.shape[1] + iPrjOffsetAOppDir),
        :-iv
    ]

# padding same direction
if zOffsetASameDir < 0:
    iv = int(-zOffsetASameDir / dv)
    paraBEx[..., :-iv] = paraA[
        :,
        (iParaAStartFinalRel + iPrjOffsetASameDir):(iParaAStartFinalRel + paraBEx.shape[1] + iPrjOffsetASameDir),
        iv:
    ]
else:
    iv = int(zOffsetASameDir / dv)
    paraBEx[..., iv:] = paraA[
        :,
        (iParaAStartFinalRel + iPrjOffsetASameDir):(iParaAStartFinalRel + paraBEx.shape[1] + iPrjOffsetASameDir),
        :-iv
    ]

# put paraB in the middle
offset = int(cba - cbb)

paraBEx[offset:offset + L, ...] = \
    paraB[:L, iParaBStartRel:iParaBEndRel, :] * (1 - w) + w * paraBEx[offset:offset + L, ...]

paraBEx[offset + paraB.shape[0] - L:offset + paraB.shape[0], ...] = \
    paraB[-L:, iParaBStartRel:iParaBEndRel, :] * w \
    + (1 - w) * paraBEx[offset + paraB.shape[0] - L:offset + paraB.shape[0], ...]

paraBEx[offset + L:offset + paraB.shape[0] - L, ...] = paraB[L:-L, iParaBStartRel:iParaBEndRel, :]


# %%
# truncate to same length
paraEx = np.array([
    paraA[:, iParaAStartFinalRel:iParaAStartFinalRel + paraBEx.shape[1], :],
    paraBEx
])[..., np.newaxis]
theta0a = anglesA[iParaAStart - iPrjAStart + iParaAStartFinalRel]
theta0b = anglesB[iParaBStart - iPrjBStart + iParaBStartRel]
z0a = posA[iParaAStart - iPrjAStart + iParaAStartFinalRel]
z0b = posB[iParaBStart - iPrjBStart + iParaBStartRel]

# %%
np.savez(
    args.out + '_' + args.set,
    prjs=paraEx,
    db=db,
    nb=nba,
    off_b=off_ba,
    rotview=rotview,
    zrot=zrot,
    theta0a=theta0a,
    theta0b=theta0b,
    z0a=z0a,
    z0b=z0b
)

# %%
import matplotlib.pyplot as plt

prjs_rebin_ab = paraEx[..., 0].transpose((0, 2, 3, 1))
print(prjs_rebin_ab.shape)
plt.figure(figsize=[16, 8])
plt.imshow(prjs_rebin_ab[0, 1000:1500, 26, :])

# %%
