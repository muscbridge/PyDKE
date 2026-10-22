import multiprocessing
import os
import os.path as op
import random
from typing import List, Self, Tuple, Union

import cvxpy as cvx
import nibabel as nib
import numpy as np
# import numpy.matlib as npm
import scipy.linalg as sla
from joblib import Parallel, delayed
from scipy.special import factorial, gamma, hyp1f1
from tqdm import tqdm

import math

from ..plotting import outlierplot
from ..system.models import input_path_validator
from ..system.utils import highprecisionexp, highprecisionpower, vectorize, writeNii
from ..tractography import dsistudio, odf, sphericalsampling

# from pydesigner.plotting import outlierplot
# from pydesigner.system.models import input_path_validator
# from pydesigner.system.utils import highprecisionexp, highprecisionpower, vectorize, writeNii
# from pydesigner.tractography import dsistudio, odf, sphericalsampling

from . import dwi_fnames, dwidirs
from . import thresholds as th

# Define the lowest number possible before it is considered a zero
minZero = th.__minZero__
# Define number of directions to resample after computing all tensors
dirSample = th.__dirs__
# Progress bar Properties
tqdmWidth = 70  # Number of columns of progress bar
# Set default numpy errorstates
defaultErrorState = np.geterr()
np.seterr(all="ignore")



class DWI(object):
    """The DWI object handles tensor estimation and parameter extraction
    of dwiffusion weighted images.

    Attributes
    ----------
    hdr: class
        Nibabel class object of input nifti file.
    img: ndarray
        3D or 4D input image array.
    grad: ndarray
        [N x 4] gradient and bvalue scheme, where the first three
        columns are the X, Y, and Z gradient vectors respectively, and
        the fourth column is B-values.
    mask: ndarray(dtype=bool)
        3D array of brain mask.
    MaskStatus: bool
        True if brain_mask.nii is present, False otherwise.
    workers: int
        Number of CPU workers to use in processing.
    """

    def __init__(
        self,
        imPath: str,
        bvecPath: str = None,
        bvalPath: str = None,
        mask: str = None,
        nthreads: int = -1,
        bvec_flips: tuple = (1, 1, 1),
    ) -> None:
        """DWI class initializer

        Parameters
        ----------
        imPath: str
            Path to input nifti file.
        bvecPath: str, optional
            Path to accompanying .bvec gradient vector file.
        bvalPath: str, optional
            Path to accompanying .bval b-value file.
        mask: str, optional
            Path to brain mask in nifti format.
        nthreads : int, optional
            Number of CPU workers to use in processing (Defaults to
            all physically present workers).
        """
        if not os.path.exists(imPath):
            msg = f"Input image ({imPath}) not found."
            raise FileNotFoundError(msg)
        self.hdr = nib.load(imPath)
        self.img = np.array(self.hdr.dataobj)
        truncateIdx = np.logical_or(np.isnan(self.img), (self.img < minZero))
        self.img[truncateIdx] = minZero
        # Get just NIFTI filename + extensio
        (path, file) = os.path.split(imPath)
        # Remove extension from NIFTI filename
        fName = os.path.splitext(file)[0]
        if bvecPath:
            input_path_validator(bvecPath, ".bvec")
        else:
            bvecPath = os.path.join(path, fName + ".bvec")
        if bvalPath:
            input_path_validator(bvalPath, ".bval")
        else:
            bvalPath = os.path.join(path, fName + ".bval")
        if os.path.exists(bvalPath) and os.path.exists(bvecPath):
            # Load bvecs
            bvecs = np.loadtxt(bvecPath)
            # Load bvals
            bvals = np.loadtxt(bvalPath)
            # Scale bvals to ms/um2 by checking for number of digits
            # in max bval
            if int(np.log10(np.max(bvals))) + 1 >= 3:  # if no. of digits >= 3
                bvals = bvals / 1000
            # Combine bvecs and bvals into [n x 4] array where n is
            # number of DWI volumes. [Gx Gy Gz Bval]
            self.grad = np.c_[np.transpose(bvecs), bvals]

            # Apply bvec sign correction.
            # Your permutation/flip test identified flip1m11 = (1, -1, 1).
            bvec_flips = np.asarray(bvec_flips, dtype=float)

            if bvec_flips.shape != (3,):
                raise ValueError(
                    "bvec_flips must be a 3-element tuple/list, e.g. (1, -1, 1)."
                )

            if not np.all(np.isin(bvec_flips, [-1, 1])):
                raise ValueError(
                    "bvec_flips values must be either 1 or -1, e.g. (1, -1, 1)."
                )

            self.grad[:, :3] *= bvec_flips[None, :]

            print(f"Applied bvec flips: {tuple(map(int, bvec_flips))}")
            print(f"bvecs, bvals (shape): {np.shape(self.grad)}")
        else:
            msg = "Unable to locate BVAL or BVEC files"
            msg += "\nPaths being used are:"
            msg += f"\nBVAL: {bvalPath}"
            msg += f"\nBVEC: {bvecPath}"
            raise OSError(msg)
        if mask is None:
            maskPath = os.path.join(path, "brain_mask.nii")
        else:
            maskPath = mask
        if os.path.exists(maskPath):
            tmp = nib.load(maskPath)
            self.mask = np.array(tmp.dataobj).astype(bool)
            self.maskStatus = True
        else:
            self.mask = np.ones((self.img.shape[0], self.img.shape[1], self.img.shape[2]), order="F")
            self.maskStatus = False
            msg = "No brain mask supplied"
            print(msg)
        tqdm.write(f"Image {fName}.nii loaded successfully")
        if nthreads is not None:
            if not isinstance(nthreads, int):
                msg = "Variable nthreads need to be an integer"
                raise TypeError(msg)
            if nthreads < -1 or nthreads == 0:
                msg = "Variable nthreads is a positive integer or -1"
                raise ValueError(msg)
        if nthreads is None:
            self.workers = -1
        else:
            self.workers = nthreads
        if self.workers == -1:
            tqdm.write("Processing with " + str(multiprocessing.cpu_count()) + " workers...")
        else:
            tqdm.write("Processing with " + str(self.workers) + " workers...")

    def getBvals(self) -> np.ndarray:
        """Returns a vector of b-values, requires no input arguments.

        Returns
        -------
        ndarray(dtype=float)
            Vector array of b-values.

        Examples
        --------
        bvals = dwi.getBvals(), where dwi is the DWI class object.
        """
        return self.grad[:, 3]

    def getBvecs(self) -> np.ndarray:
        """Returns an array of gradient vectors, requires no input
        parameters.

        Returns
        -------
        ndarray(dtype=float)
            [N x 3] array of gradient vectors.

        Examples
        --------
        bvecs = dwi.getBvecs(), where dwi is the DWI class object.
        """
        return self.grad[:, 0:3]

    def maxBval(self) -> float:
        """Returns the maximum b-value in a dataset to determine between
        DTI and DKI, requires no input parameters.

        Returns
        -------
        float
            Maximum B-value in DWI.

        Examples
        --------
        a = dwi.maxBval(), where dwi is the DWI class object.

        """
        return max(np.unique(self.grad[:, 3]))

    def maxDTIBval(self) -> float:
        """Returns the maximum DTI b-value in a dataset.

        Returns
        -------
        float
            Maximum DTI B-value in DWI.

        Examples
        --------
        a = dwi.maxDKIBval(), where dwi is the DWI class object

        """
        exclude_idx = self.grad[:, 3] <= th.__maxdtibval__
        return max(np.unique(self.grad[exclude_idx, 3]))

    def maxDKIBval(self) -> float:
        """Returns the maximum DKI b-value in a dataset.

        Returns
        -------
        float
            Maximum DKI B-value in DWI.

        Examples
        --------
        a = dwi.maxDKIBval(), where dwi is the DWI class object

        """
        exclude_idx = self.grad[:, 3] <= th.__maxdkibval__
        return max(np.unique(self.grad[exclude_idx, 3]))

    def maxFBIBval(self) -> float:
        """Returns the maximum FBI b-value in a dataset.

        Returns
        -------
        float
            Maximum DKI B-value in DWI.

        Examples
        --------
        a = dwi.maxDKIBval(), where dwi is the DWI class object

        """
        exclude_idx = self.grad[:, 3] <= th.__maxfbibval__
        return max(np.unique(self.grad[exclude_idx, 3]))

    def idxb0(self) -> np.ndarray[bool]:
        """Returns the index of all B-zeros according to bvals
        in record.

        Returns
        -------
        idx: ndarray(dtype=bool)
            Index of DTI/DKI b-values.

        """
        return np.rint(self.grad[:, 3]) == 0

    def idxdti(self) -> np.ndarray[bool]:
        """Returns the index of all DTI/DKI B-values according to bvals
        in record.

        Returns
        -------
        idx: ndarray(dtype=bool)
            Index of DTI/DKI b-values.

        """
        idx = np.ones_like(self.grad[:, -1], dtype=bool)
        if self.isdti():
            idx = self.grad[:, 3] <= self.maxDTIBval()
        return idx

    def idxdki(self) -> np.ndarray[bool]:
        """Returns the index of all DTI/DKI B-values according to bvals
        in record

        Returns
        -------
        idx: ndarray(dtype=bool)
            Index of DTI/DKI b-values

        """
        idx = np.ones_like(self.grad[:, -1], dtype=bool)
        if self.isdki():
            idx = self.grad[:, 3] <= self.maxDKIBval()
        return idx

    def idxfbi(self) -> np.ndarray[int]:
        """Returns the index of all FBI B-values according to bvals
        in record

        Returns
        -------
        idx: ndarray(dtype=bool)
            Index of DTI/DKI b-values.

        """
        idx = np.ones_like(self.grad[:, 3], dtype=bool)
        if self.isfbi():
            idx = np.rint(self.grad[:, -1]) == np.rint(self.maxFBIBval())
        else:
            raise IndexError("No valid FBI sequence found.")
        return idx

    def getndirs(self) -> int:
        """Returns the number of gradient directions acquired from the
        scanner

        Returns
        -------
        int
            Number of gradient directions

        Examples
        --------
        n = dwi.getndirs(), where dwi is the DWI class object
        """
        return np.sum(self.grad[:, 3] == self.maxDKIBval())

    def tensorType(self) -> List[str]:
        """Returns whether input image is DTI or DKI compatible, requires
        no input parameters

        Returns
        -------
        type: list[str]
            Contains list of string 'dti', 'dki', or 'fbi' based on
            the protocols the input DWI represents

        Examples
        --------
        a = dwi.tensorType(), where dwi is the DWI class object
        """
        type = []
        if self.maxDTIBval() <= th.__maxdtibval__ and self.maxDTIBval() >= th.__mindkibval__:
            type.append("dti")
        if self.maxDKIBval() >= th.__maxdtibval__ and self.maxDKIBval() <= th.__maxdkibval__:
            type.append("dki")
        if self.maxBval() >= th.__minfbibval__:
            type.append("fbi")
        if "fbi" in type and "dki" in type:
            type.append("fbwm")
        if not type:
            raise ValueError("tensortype: Error in determining maximum BVAL")
        return type

    def isdti(self) -> bool:
        """Returns logical value to answer the mystical question whether
        the input image is DTI.

        Returns
        -------
        ans: bool
            True if DTI; false otherwise.

        Examples
        --------
        ans = dwi.isdki(), where dwi is the DWI class object.
        """
        if "dti" in self.tensorType():
            ans = True
        else:
            ans = False
        return ans

    def isdki(self) -> bool:
        """Returns logical value to answer the mystical question whether
        the input image is DKI.

        Returns
        -------
        ans: bool
            True if DKI; false otherwise.

        Examples
        --------
        ans = dwi.isdki(), where dwi is the DWI class object.
        """
        if "dki" in self.tensorType():
            ans = True
        else:
            ans = False
        return ans

    def isfbi(self) -> bool:
        """Returns bool value to specify whether image input image is
        FBI.

        Returns
        -------
        ans: bool
            True if FBI; false otherwise.

        Examples
        --------
        ans = dwi.isfbi(), where dwi is the DWI class object.
        """
        if "fbi" in self.tensorType():
            ans = True
        else:
            ans = False
        return ans

    def isfbwm(self) -> bool:
        """Returns bool value to specify whether image input image is
        FBWM.

        Returns
        -------
        and: bool
            True if FBWM; false otherwise.

        Examples
        --------
        ans = dwi.isfbi(), where dwi is the DWI class object.
        """
        if "fbwm" in self.tensorType():
            ans = True
        else:
            ans = False
        return ans

    def createTensorOrder(self, order: Union[int, None] = None) -> Tuple[np.ndarray[int], np.ndarray[int]]:
        """Creates tensor order array and indices.

        Parameters
        ----------
        order:  2 or 4 (int or None)
            Tensor order number, 2 for diffusion and 4 for kurtosis.
            Default: None; auto-detect

        Returns
        -------
        cnt: ndarray(dtype=int)
            Count of number of times a tensor appears.
        ind: ndarray(dtype=int)
            Indices of count.

        Examples
        --------
        (cnt, ind) = dwi.createTensorOrder(order)

        Notes
        -----
        The tensors for this pipeline are based on NYU's designer layout as
        depicted in the table below. This will soon be depreciated and
        updated with MRTRIX3's layout.

         .. code-block:: none

            ~~~~~~D~~~~~~
            1  |    D11
            2  |    D12
            3  |    D13
            4  |    D22
            5  |    D23
            6  |    D33
            ~~~~~~K~~~~~~
            1  |   W1111
            2  |   W1112
            3  |   W1113
            4  |   W1122
            5  |   W1123
            6  |   W1133
            7  |   W1222
            8  |   W1223
            9  |   W1233
            10 |   W1333
            11 |   W2222
            12 |   W2223
            13 |   W2233
            14 |   W2333
            15 |   W3333
        """
        if order is None:
            if self.isdki():
                cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
                ind = (
                    np.array(
                        (
                            [1, 1, 1, 1],
                            [1, 1, 1, 2],
                            [1, 1, 1, 3],
                            [1, 1, 2, 2],
                            [1, 1, 2, 3],
                            [1, 1, 3, 3],
                            [1, 2, 2, 2],
                            [1, 2, 2, 3],
                            [1, 2, 3, 3],
                            [1, 3, 3, 3],
                            [2, 2, 2, 2],
                            [2, 2, 2, 3],
                            [2, 2, 3, 3],
                            [2, 3, 3, 3],
                            [3, 3, 3, 3],
                        )
                    )
                    - 1
                )
            else:
                cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
                ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
        elif order == 2:
            cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
            ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
        elif order == 4:
            cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
            ind = (
                np.array(
                    (
                        [1, 1, 1, 1],
                        [1, 1, 1, 2],
                        [1, 1, 1, 3],
                        [1, 1, 2, 2],
                        [1, 1, 2, 3],
                        [1, 1, 3, 3],
                        [1, 2, 2, 2],
                        [1, 2, 2, 3],
                        [1, 2, 3, 3],
                        [1, 3, 3, 3],
                        [2, 2, 2, 2],
                        [2, 2, 2, 3],
                        [2, 2, 3, 3],
                        [2, 3, 3, 3],
                        [3, 3, 3, 3],
                    )
                )
                - 1
            )
        else:
            raise ValueError("createTensorOrder: Please enter valid order values (2 or 4).")
        return cnt, ind

    def fibonacciSphere(self, samples: int = 1, randomize: bool = True) -> np.ndarray[float]:
        """Returns evenly spaced points on a sphere.

        Parameters
        ----------
        samples: int
            Number of points to compute from sphere, must be a
            positive and real integer (Default: 1).

        randomize: bool
                    True if sampling is randomized; False otherwise
                    (Default: True).

        Returns
        -------
        points: ndarray(dtype=float)
            [3 x samples] array containing evenly spaced points
            from a sphere.

        Examples
        --------
        dirs = dwi.fibonacciSphere(256, True)
        """
        rnd = 1
        if randomize:
            rnd = random.random() * samples
        points = []
        offset = 2 / samples
        increment = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y, 2))
            phi = ((i + rnd) % samples) * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x, y, z])
        return np.array(points)

    def radialSampling(self, dir: np.ndarray[float], n: int) -> np.ndarray[float]:
        """Get the radial component of a metric from a set of directions.

        Parameters
        ----------
        dir: ndarray(dtype=float)
            [n x 3] input array of directions.
        n: int
            Number of rows, n.

        Returns
        -------
        dirs: ndarray(dtype=float)
            Matrix containing radial components.

        Examples
        --------
        grad = dwi.radialSampling(dir, number_of_dirs).

        """
        dt = 2 * np.pi / n
        theta = np.arange(0, 2 * np.pi - dt, dt)
        dirs = np.vstack((np.cos(theta), np.sin(theta), 0 * theta))
        v = np.hstack((-dir[1], dir[0], 0))
        s = np.sqrt(np.sum(v**2))
        c = dir[2]
        V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + V + np.matmul(V, V) * (1 - c) / (s**2)
        dirs = np.matmul(R, dirs).T
        return dirs

    def diffusionCoeff(self, dt: np.ndarray[float], dir: np.ndarray[float]) -> np.ndarray[float]:
        """Computes apparent diffusion coefficient (ADC).

        Parameters
        ----------
        ----------S
        dt: ndarray(dtype=float)
            [21 x nvoxel] array containing diffusion tensor.
        dir:  ndarray(dtype=float)
            [n x 3] array containing gradient directions.

        Returns
        -------
        adc : ndarray(dtype=float)
            Array containing apparent diffusion coefficient.

        Examples
        --------
        adc = dwi.diffusionCoeff(dt, dir)
        """
        dcnt, dind = self.createTensorOrder(2)
        ndir = dir.shape[0]
        bD = np.tile(dcnt, (ndir, 1)) * dir[:, dind[:, 0]] * dir[:, dind[:, 1]]
        adc = np.matmul(bD, dt)
        return adc

    def kurtosisCoeff(self, dt: np.ndarray[float], dir: np.ndarray[float]) -> np.ndarray[float]:
        """Computes apparent kurtosis coefficient (AKC)

        Parameters
        ----------
        dt : ndarray(dtype=float)
            [21 x nvoxel] array containing diffusion tensor
        dir : ndarray(dtype=float)
            [n x 3] array containing gradient directions

        Returns
        -------
        adc : ndarray(dtype=float)
            Array containing apparent kurtosis coefficient

        Examples
        --------
        adc = dwi.kurtosisCoeff(dt, dir)
        """
        wcnt, wind = self.createTensorOrder(4)
        ndir = dir.shape[0]
        adc = self.diffusionCoeff(dt[:6], dir)
        adc[adc < minZero] = minZero
        md = np.sum(dt[np.array([0, 3, 5])], 0) / 3
        bW = (
            np.tile(wcnt, (ndir, 1)) * dir[:, wind[:, 0]] * dir[:, wind[:, 1]] * dir[:, wind[:, 2]] * dir[:, wind[:, 3]]
        )
        akc = np.matmul(bW, dt[6:])
        akc = (akc * np.tile(md**2, (adc.shape[0], 1))) / (adc**2)
        return akc

    def dtiTensorParams(self, dt: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """Computes sorted DTI tensor eigenvalues and eigenvectors.

        Parameters
        ----------
        DT: ndarray(dtype=float)
            Diffusion tensor array.

        Returns
        -------
        values: ndarray(dtype=float)
            Array of sorted eigenvalues.
        vectors: ndarray(dtype=float)
            Array lf sorted eigenvectors.

        Examples
        --------
        (values, vectors) = dwi.dtiTensorParams(dt)
        """
        # Diffusion tensors are real symmetric; eigh keeps eigenvectors real
        # across NumPy versions and avoids complex-valued scalar maps.
        values, vectors = np.linalg.eigh(dt)
        idx = np.argsort(-values)
        values = -np.sort(-values)
        vectors = vectors[:, idx]
        return values, vectors

    def dkiTensorParams(
        self, v1: np.ndarray[float], dt: np.ndarray[float]
    ) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """Uses average directional statistics to approximate axial
        kurtosis(AK) and radial kurtosis (RK).

        Parameters
        ----------
        v1 : ndarray(dtype=float)
            Array of first eigenvectors from DWI.dtiTensorParams().
        dt : ndarray(dtype=float)
            Array of diffusion tensor.

        Returns
        -------
        rk : ndarray(dtype=float)
            Radial Kurtosis.
        ak : ndarray(dtype=float)
            Axial Kurtosis.
        kfa : ndarray(dtype=float)
            Kurtosis Fractional Anisotropy.
        mkt : ndarray(dtype=float)
            Mean Kurtosis Tensor.

        Examples
        --------
        (rk, ak, kfa, mkt) = dwi.dkiTensorParams(v1, dt)
        """
        dirs = np.vstack((v1, -v1))
        akc = self.kurtosisCoeff(dt, dirs)
        ak = np.mean(akc)
        dirs = self.radialSampling(v1, dirSample)
        akc = self.kurtosisCoeff(dt, dirs)
        rk = np.mean(akc)
        W_F = np.sqrt(
            dt[6] ** 2
            + dt[16] ** 2
            + dt[20] ** 2
            + 6 * dt[9] ** 2
            + 6 * dt[11] ** 2
            + 6 * dt[18] ** 2
            + 4 * dt[7] ** 2
            + 4 * dt[8] ** 2
            + 4 * dt[12] ** 2
            + 4 * dt[17] ** 2
            + 4 * dt[15] ** 2
            + 4 * dt[19] ** 2
            + 12 * dt[10] ** 2
            + 12 * dt[13] ** 2
            + 12 * dt[14] ** 2
        )
        Wbar = 1 / 5 * (dt[6] + dt[16] + dt[20] + 2 * (dt[9] + dt[11] + dt[18]))
        if W_F < minZero:
            kfa = 0
        else:
            W_diff = np.sqrt(
                (dt[6] - Wbar) ** 2
                + (dt[16] - Wbar) ** 2
                + (dt[20] - Wbar) ** 2
                + 6 * (dt[9] - Wbar / 3) ** 2
                + 6 * (dt[11] - Wbar / 3) ** 2
                + 6 * (dt[18] - Wbar / 3) ** 2
                + 4 * dt[7] ** 2
                + 4 * dt[8] ** 2
                + 4 * dt[12] ** 2
                + 4 * dt[17] ** 2
                + 4 * dt[15] ** 2
                + 4 * dt[19] ** 2
                + 12 * dt[10] ** 2
                + 12 * dt[13] ** 2
                + 12 * dt[14] ** 2
            )
            kfa = W_diff / W_F
        mkt = Wbar
        return ak, rk, kfa, mkt

    def wlls(
        self,
        shat: np.ndarray,
        dwi: np.ndarray,
        b: np.ndarray,
        cons: Union[np.ndarray, None] = None,
        warmup: Union[np.ndarray, None] = None,
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """Weighted linear least squares for voxelwise DT/DKT fitting.

        With return_diagnostics=True, return (coefficients, diagnostics),
        exposing solver status, unconstrained fallback, signal flooring and
        constraint residuals. The default array return remains unchanged.
        Constraint margins use raw fitted coefficients before MD normalization.
        Constrained objectives are normalized by a common positive weight scale.
        A solver success status alone is insufficient: sampled inequality margins
        are checked. Invalid constrained results raise unless diagnostics were
        explicitly requested, in which case valid_constrained_solution is False.
        """

        # Make inputs safe
        shat = np.asarray(shat, dtype=np.float64).reshape(-1)
        dwi = np.asarray(dwi, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64)

        diagnostics = {
            "solver_status": "not_requested" if cons is None else "not_run",
            "solver_iterations": None,
            "fallback_used": False,
            "fallback_reason": None,
            "signal_samples_floored": int(np.sum(~np.isfinite(dwi) | (dwi < minZero))),
            "nonfinite_solution_coefficients": 0,
            "minimum_constraint_margin": None,
            "constraint_margin_tolerance": 1e-6,
            "valid_constrained_solution": None,
            "weight_normalization_factor": 1.0,
        }

        def result(coefficients, sanitize=False):
            diagnostics["nonfinite_solution_coefficients"] = int(np.sum(~np.isfinite(coefficients)))
            if sanitize:
                coefficients = np.nan_to_num(coefficients, nan=minZero, posinf=minZero, neginf=minZero)
            if cons is not None and len(cons) and np.isfinite(coefficients).all():
                diagnostics["minimum_constraint_margin"] = float(np.min(cons @ coefficients))
            if cons is not None:
                margin = diagnostics["minimum_constraint_margin"]
                diagnostics["valid_constrained_solution"] = bool(
                    not diagnostics["fallback_used"]
                    and diagnostics["solver_status"] in ("optimal", "optimal_inaccurate")
                    and diagnostics["nonfinite_solution_coefficients"] == 0
                    and margin is not None
                    and margin >= -diagnostics["constraint_margin_tolerance"]
                )
                if not diagnostics["valid_constrained_solution"] and not return_diagnostics:
                    raise RuntimeError(
                        "Constrained WLLS did not produce a valid sampled-constraint solution; "
                        "request return_diagnostics=True to inspect status, fallback and residuals"
                    )
            return (coefficients, diagnostics) if return_diagnostics else coefficients

        # Avoid log(0), negative signal, NaN, Inf
        dwi = np.nan_to_num(dwi, nan=minZero, posinf=minZero, neginf=minZero)
        dwi[dwi < minZero] = minZero

        shat = np.nan_to_num(shat, nan=0.0, posinf=0.0, neginf=0.0)
        shat = np.clip(shat, 0.0, 1e6)
        if cons is not None:
            scale = float(np.max(shat))
            diagnostics["weight_normalization_factor"] = scale
            if scale <= 0:
                diagnostics["solver_status"] = "invalid_weights"
                return result(np.full(b.shape[1], np.nan), sanitize=True)
            # A common positive factor leaves the exact WLS minimizer unchanged.
            # It prevents absolute MRI intensity from relaxing the relative QP
            # stopping criterion for the much smaller physical constraint rows.
            shat = shat / scale

        w = np.diag(shat)

        # Safe unconstrained fallback
        def unconstrained_solution():
            with np.errstate(all="ignore"):
                C_u = np.matmul(w, b).astype(np.float64)
                d_u = np.matmul(w, np.log(dwi)).astype(np.float64)
            C_u = np.nan_to_num(C_u, nan=0.0, posinf=0.0, neginf=0.0)
            d_u = np.nan_to_num(d_u, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                dt_u = np.matmul(np.linalg.pinv(C_u), d_u)
            except np.linalg.LinAlgError:
                dt_u = np.full(b.shape[1], np.nan)
            return dt_u

        # Unconstrained fitting
        if cons is None:
            return result(unconstrained_solution())

        # Constrained fitting
        with np.errstate(all="ignore"):
            C = np.matmul(w, b).astype(np.float64)
            d = np.matmul(w, np.log(dwi)).astype(np.float64).reshape(-1)

        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        m, n = C.shape
        x = cvx.Variable(n)

        if warmup is not None:
            try:
                x.value = np.asarray(warmup, dtype=np.float64).reshape(-1)
            except Exception:
                pass

        objective = cvx.Minimize(0.5 * cvx.sum_squares(C @ x - d))
        constraints = [cons @ x >= np.zeros(len(cons))]
        prob = cvx.Problem(objective, constraints)

        try:
            with np.errstate(all="ignore"):
                prob.solve(
                    solver=cvx.OSQP,
                    warm_start=True,
                    max_iter=200000,
                    eps_abs=1e-9,
                    eps_rel=1e-9,
                    polishing=True,
                    verbose=False,
                )

            diagnostics["solver_status"] = prob.status
            if prob.solver_stats is not None and prob.solver_stats.num_iters is not None:
                diagnostics["solver_iterations"] = int(prob.solver_stats.num_iters)
            # Accept lower-accuracy successful solves too
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                dt = np.asarray(x.value, dtype=np.float64).reshape(-1)
                # print(f"prob.status = {prob.status}")
            else:
                diagnostics["fallback_used"] = True
                diagnostics["fallback_reason"] = "solver_status: " + str(prob.status)
                dt = unconstrained_solution()

        except (FloatingPointError, ValueError, cvx.error.SolverError) as e:
            diagnostics["solver_status"] = "error"
            diagnostics["fallback_used"] = True
            diagnostics["fallback_reason"] = type(e).__name__ + ": " + str(e)
            dt = unconstrained_solution()

        return result(dt, sanitize=True)

    def fit(self, constraints: Union[np.ndarray[float], None] = None, reject: bool = None) -> Self:
        """Returns fitted diffusion or kurtosis tensor

        Parameters
        ----------
        constraints: array_like(dtype=int)
            [1 x 3] vector that specifies which constraints to use
            (Default: None)
        reject: ndarray(dtype=bool)
            4D array containing information on voxels to exclude
            from DT estimation (Default: None)

        Examples
        --------
        dwi.fit()
        dwi.fit(constraints=[0,1,0], reject=irlls_output)
        """
        # Handle rejected voxels from IRLLS
        exclude_idx = self.idxdki()
        if reject is None:
            reject = np.zeros(self.img[:, :, :, exclude_idx].shape)
        grad = self.grad[exclude_idx, :]
        grad_orig = grad.copy()
        order = np.floor(np.log(np.abs(np.max(grad[:, -1]) + 1)) / np.log(10))
        img = self.img[:, :, :, exclude_idx]
        if order >= 2:
            grad[:, -1] = grad[:, -1]
        img.astype(np.double)
        img[img <= 0] = np.finfo(np.double).eps
        grad.astype(np.double)
        normgrad = np.sqrt(np.sum(grad[:, :3] ** 2, 1))
        normgrad[normgrad == 0] = 1
        grad[:, :3] = grad[:, :3] / np.tile(normgrad, (3, 1)).T
        grad[np.isnan(grad)] = 0
        dcnt, dind = self.createTensorOrder(2)
        wcnt, wind = self.createTensorOrder(4)
        ndwis = img.shape[-1]
        bs = np.ones((ndwis, 1))
        bD = np.tile(dcnt, (ndwis, 1)) * grad[:, dind[:, 0]] * grad[:, dind[:, 1]]
        bW = (
            np.tile(wcnt, (ndwis, 1))
            * grad_orig[:, wind[:, 0]]
            * grad_orig[:, wind[:, 1]]
            * grad_orig[:, wind[:, 2]]
            * grad_orig[:, wind[:, 3]]
        )
        self.b = np.concatenate(
            (
                bs,
                (np.tile(-self.grad[exclude_idx, -1], (6, 1)).T * bD),
                np.squeeze(1 / 6 * np.tile(self.grad[exclude_idx, -1], (15, 1)).T ** 2) * bW,
            ),
            1,
        )
        dwi_ = vectorize(img, self.mask)
        reject_ = vectorize(reject, self.mask).astype(bool)
        init = np.matmul(np.linalg.pinv(self.b), np.log(dwi_))
        shat = highprecisionexp(np.matmul(self.b, init))
        if constraints is None or (constraints[0] == 0 and constraints[1] == 0 and constraints[2] == 0):
            inputs = tqdm(
                range(0, dwi_.shape[1]),
                desc="Unconstrained Tensor Fit",
                bar_format="{desc}: [{percentage:0.0f}%]",
                unit="vox",
                ncols=tqdmWidth,
            )
            self.dt = Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(self.wlls)(
                    shat[~reject_[:, i], i],
                    dwi_[~reject_[:, i], i],
                    self.b[~reject_[:, i]],
                )
                for i in inputs
            )
        else:
            # C is linear inequality constraint matrix A_ub
            C = self.createConstraints(constraints)
            inputs = tqdm(
                range(0, dwi_.shape[1]),
                desc="Constrained Tensor Fit",
                bar_format="{desc}: [{percentage:0.0f}%]",
                unit="vox",
                ncols=tqdmWidth,
            )
            self.dt = Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(self.wlls)(
                    shat[~reject_[:, i], i],
                    dwi_[~reject_[:, i], i],
                    self.b[~reject_[:, i]],
                    cons=C,
                )
                for i in inputs
            )
        self.dt = np.reshape(self.dt, (dwi_.shape[1], self.b.shape[1])).T
        self.s0 = highprecisionexp(self.dt[0, :])
        self.dt = self.dt[1:, :]
        D_apprSq = 1 / (np.sum(self.dt[(0, 3, 5), :], axis=0) / 3) ** 2
        self.dt[6:, :] = self.dt[6:, :] * np.tile(D_apprSq, (15, 1))

    def createConstraints(self, constraints: List[int] = [0, 1, 0]) -> np.ndarray[float]:
        """Generates constraint array for constrained minimization quadratic
        programming.

        Parameters
        ----------
        constraints:   array_like(dtype=int)
            [1 X 3] logical vector indicating which constraints
            out of three to enable (Default: [0, 1, 0])
            C1 is Dapp > 0
            C2 is Kapp > 0
            C3 is Kapp < 3/(b*Dapp)

        Returns
        -------
        C: ndarray(dtype=float)
            Array containing constraints to consider during
            minimization, C is shaped [number of constraints enforced *
            number of directions, 22].

        Examples
        --------
        C = dwi.createConstraints([0, 1, 0])
        """
        if sum(constraints) >= 0 and sum(constraints) <= 3:
            dcnt, dind = self.createTensorOrder(2)
            wcnt, wind = self.createTensorOrder(4)
            cDirs = dwidirs.dirs30
            ndirs = cDirs.shape[0]
            C = np.empty((0, 22))
            if constraints[0] > 0:  # Dapp > 0
                C = np.append(
                    C,
                    np.hstack(
                        (
                            np.zeros((ndirs, 1)),
                            np.tile(dcnt, [ndirs, 1]) * cDirs[:, dind[:, 0]] * cDirs[:, dind[:, 1]],
                            np.zeros((ndirs, 15)),
                        )
                    ),
                    axis=0,
                )
            if constraints[1] > 0:  # Kapp > 0
                C = np.append(
                    C,
                    np.hstack(
                        (
                            np.zeros((ndirs, 7)),
                            np.tile(wcnt, [ndirs, 1])
                            * cDirs[:, wind[:, 0]]
                            * cDirs[:, wind[:, 1]]
                            * cDirs[:, wind[:, 2]]
                            * cDirs[:, wind[:, 3]],
                        )
                    ),
                    axis=0,
                )
            if constraints[2] > 0:  # K < 3/(b*Dapp)
                C = np.append(
                    C,
                    np.hstack(
                        (
                            np.zeros((ndirs, 1)),
                            # log(S) = log(S0) - b*Dapp + b^2*Wraw/6.
                            # Monotonicity through bmax requires
                            # (3/bmax)*Dapp - Wraw >= 0, with complete
                            # quadratic and quartic directional contractions.
                            3 / self.maxDKIBval() * np.tile(dcnt, [ndirs, 1])
                            * cDirs[:, dind[:, 0]] * cDirs[:, dind[:, 1]],
                            np.tile(-wcnt, [ndirs, 1])
                            * cDirs[:, wind[:, 0]]
                            * cDirs[:, wind[:, 1]]
                            * cDirs[:, wind[:, 2]]
                            * cDirs[:, wind[:, 3]],
                        )
                    ),
                    axis=0,
                )
        else:
            msg = "Invalid contraints. Please use format [0, 0, 0]"
            raise ValueError(msg)
        return C

    def extractDTI(
        self,
    ) -> Tuple[
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
    ]:
        """Extract all DTI parameters from DT tensor. Warning, this can
        only be run after tensor fitting dwi.fit().

        Returns
        -------
        md: ndarray(dtype=float)
            Mean Diffusivity.
        rd: ndarray(dtype=float)
            Radial Diffusivity.
        ad: ndarray(dtype=float)
            Axial Diffusivity.
        fa: ndarray(dtype=float)
            Fractional Anisotropy.
        fe: ndarray(dtype=float)
            First Eigenvectors.
        trace: ndarray(dtype=float)
            Sum of first eigenvalues.

        Examples
        --------
        (md, rd, ad, fa) = dwi.extractDTI(), where dwi is the DWI class
        object
        """

        # print("extractDTI self.dt shape:", self.dt.shape)
        # print("extractDTI voxelwise std (axis=1):", np.std(self.dt, axis=1))
        # print("extractDTI first 5 voxels (first 6 coeffs):\n", self.dt[:6, :5])

        # extract all tensor parameters from dt
        DT = np.reshape(
            np.concatenate(
                (
                    self.dt[0, :],
                    self.dt[1, :],
                    self.dt[2, :],
                    self.dt[1, :],
                    self.dt[3, :],
                    self.dt[4, :],
                    self.dt[2, :],
                    self.dt[4, :],
                    self.dt[5, :],
                )
            ),
            (3, 3, self.dt.shape[1]),
        )
        # get the trace
        rdwi = highprecisionexp(np.matmul(self.b[:, 1:], self.dt))
        B = np.round(-(self.b[:, 0] + self.b[:, 3] + self.b[:, 5]) * 1000)
        uB = np.unique(B)
        trace = np.zeros((self.dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]):
            t = np.where(B == uB[ib])
            trace[:, ib] = np.mean(rdwi[t[0], :], axis=0)
        nvox = self.dt.shape[1]
        inputs = tqdm(
            range(0, nvox),
            desc="DTI Parameters",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="vox",
            ncols=tqdmWidth,
        )
        values, vectors = zip(
            *Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(self.dtiTensorParams)(DT[:, :, i]) for i in inputs
            )
        )
        values = np.reshape(np.abs(values), (nvox, 3))
        vectors = np.reshape(vectors, (nvox, 3, 3))
        self.DTIvectors = vectors
        l1 = vectorize(values[:, 0], self.mask)
        l2 = vectorize(values[:, 1], self.mask)
        l3 = vectorize(values[:, 2], self.mask)
        v1 = vectorize(vectors[:, :, 0].T, self.mask)
        md = (l1 + l2 + l3) / 3
        rd = (l2 + l3) / 2
        ad = l1
        fa = np.sqrt(1 / 2) * np.sqrt((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / np.sqrt(l1**2 + l2**2 + l3**2)
        fe = np.abs(np.stack((fa * v1[:, :, :, 0], fa * v1[:, :, :, 1], fa * v1[:, :, :, 2]), axis=3))
        trace = vectorize(trace.T, self.mask)
        return md, rd, ad, fa, fe, trace

    def extractDKI(
        self,
    ) -> Tuple[
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
    ]:
        """Extract all DKI parameters from DT tensor. Warning, this can
        only be run after tensor fitting dwi.fit().

        Returns
        -------
        mk: ndarray(dtype=float)
            Mean Diffusivity.
        rk: ndarray(dtype=float)
            Radial Diffusivity.
        ak: ndarray(dtype=float)
            Axial Diffusivity.
        kfa: ndarray(dtype=float)
            Kurtosis Fractional Anisotropy.
        mkt: ndarray(dtype=float)
            Mean Kurtosis Tensor.
        trace: ndarray(dtype=float)
            Sum of first eigenvalues.

        Examples
        --------
        (mk, rk, ak, fe, trace) = dwi.extractDTI(), where dwi is the DWI
        class object
        """
        # get the trace
        rdwi = highprecisionexp(np.matmul(self.b[:, 1:], self.dt))
        B = np.round(-(self.b[:, 0] + self.b[:, 3] + self.b[:, 5]) * 1000)
        uB = np.unique(B)
        trace = np.zeros((self.dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]):
            t = np.where(B == uB[ib])
            trace[:, ib] = np.mean(rdwi[t[0], :], axis=0)
        dirs = dwidirs.dirs256
        akc = self.kurtosisCoeff(self.dt, dirs)
        mk = np.mean(akc, 0)
        nvox = self.dt.shape[1]
        inputs = tqdm(
            range(0, nvox),
            desc="DKI Parameters",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="vox",
            ncols=tqdmWidth,
        )
        ak, rk, kfa, mkt = zip(
            *Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(self.dkiTensorParams)(self.DTIvectors[i, :, 0], self.dt[:, i]) for i in inputs
            )
        )
        ak = np.reshape(ak, (nvox))
        rk = np.reshape(rk, (nvox))
        kfa = np.reshape(kfa, (nvox))
        mkt = np.reshape(mkt, (nvox))
        trace = vectorize(trace.T, self.mask)
        ak = vectorize(ak, self.mask)
        rk = vectorize(rk, self.mask)
        mk = vectorize(mk, self.mask)
        kfa = vectorize(kfa, self.mask)
        mkt = vectorize(mkt, self.mask)
        return mk, rk, ak, kfa, mkt, trace

    def optimal_lmax(self) -> int:
        """Computes the highest harmonic order (l_max) for
        spherical harmonic expansion. This is adapted
        from the information posted at
        https://mrtrix.readthedocs.io/en/dev/concepts/sh_basis_lmax.html

        This function runs successfully only if input
        DWI is an FBI or HARDI acquisition.

        Returns
        -------
        int
            l_max suitable for DWI.
        """
        if not self.isfbi():
            raise Exception("Input DWI is not an FBI or HARDI acquisiton. Cannot compute l_max.")
        fbi_vols = np.count_nonzero(self.grad[self.idxfbi(), -1])
        l_max = 0
        vols = (l_max + 1) * (l_max / 2 + 1)
        while vols <= fbi_vols:
            l_max += 2
            vols = (l_max + 1) * (l_max / 2 + 1)
        return l_max - 2

    def fbi(
        self, l_max: int = 6, fbwm: bool = True, rectify: bool = True, res: str = "med"
    ) -> Tuple[
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
    ]:
        """Perform fiber ball imaging (FBI) and FBI white matter model
        (FBWM) analyses

        Parameters
        ----------
        l_max: int
            Maximum spherical harmonic degree specified as an even
            integer
            (Default: 6)
        fbwm: bool
            Perform FBWM parameterization if True
            (Default: True)
        rectify: bool
            Perform fODF rectification if True
            (Default: True)
        res: str
            Resolution of spherical sampling distribution (Default: 'med')
            'low' defines the spherical grid defined by 3 fold quadrisection of the
            isocahedron, or 8 fold tesselation of icosahedron.
            'med' defines the spherical grid defined by 4 fold quadrisection of the
            isocahedron, or 16 fold tesselation of icosahedron.
            'high' defines the spherical grid defined by 5 fold quadrisection of the
            isocahedron, or 24 fold tesselation of icosahedron.
            Default: "med"

        Returns
        -------
        zeta: array_like(dtype=float)
            Zeta parameter
        faa: array_like(dtype=float)
            Intra-axonal fractional anisotropy
        fodf: array_like(dtype=float)
            fodf from spherical harmonic expansion
        min_awf: array_like(dtype=float)
            Axonal water fraction
        Da: array_like(dtype=float)
            Intrinsic intra-axonal diffusivity
        De_mean: array_like(dtype=float)
            Mean extra-axonal diffusion
        De_ax: array_like(dtype=float)
            Axial extra-axonal diffusion
        De_rad: array_like(dtype=float)
            Radial extra-axonal diffusion
        De_fa: array_like(dtype=float)
            Extra-axonal FA
        min_cost: array_like(dtype=float)
            Minimum cost of the cost function (first index of
            min_cost_fn)
        min_cost_fn: array_like(dtype=float)
            Cost function
        """
        # --------------------FUNCTION SEPARATOR-----------------------

        def __fbi_rectify(fodf, sh_area, iter=1000):
            """Rectifies fODF values to eliminate all negative values while
            reducing the mean square error

            Parameters
            ----------
            fodf : array_like(dtype=float)
                Real portion of fODF
            iter : int
                Number of iterations to perform
                (Default: 1000)
            sh_area: array_like(dtype=float)
                Area of spherical sampling

            Returns
            -------
            odf : float
                Rectified fODF
            """
            # fODF rectification
            odf = fodf
            fODF = fodf.real  # grab real part of the fODF
            fODF[np.isnan(fODF)] = 0
            Fmax = np.max(fODF)  # get the max peak value of the ODF
            lB = 0  # initial lower bound
            uB = Fmax  # initial upper bound
            M = 1  # initialze iteration counter
            Mmax = iter  # max iterations (could probably be 100 too)
            if Fmax > 0:
                while M <= Mmax:
                    # BEGIN: bi-section algorithm
                    midpt = (lB + uB) / 2
                    fODF_lB = np.sum((np.abs(fODF - lB) - fODF - lB) * sh_area, axis=0)
                    fODF_midpt = np.sum((np.abs(fODF - midpt) - fODF - midpt) * sh_area, axis=0)
                    if fODF_midpt == 0 or (uB - lB) / 2 < minZero:
                        EPS = midpt
                        break
                    else:
                        M = M + 1
                        if np.sign(fODF_midpt) == np.sign(fODF_lB):
                            lB = midpt
                        else:
                            uB = midpt
                    # END: bi-section algorithm
                # Subract solution from each ODF point
                odf = (1 / 2) * (np.abs(odf - EPS) + odf - EPS)
                odf = odf.real
                # due to numerical error, we manually set
                # very very very tiny peaks to zero after the fact...
                odf[np.logical_and(odf > -minZero, odf < minZero)] = 0
            return odf

        def __costCalculator(grid, BT, GT, b0, IMG, iDT, iaDT, zeta, shB, Pl0, g2l_fa_R_b, clm):
            """Computes the cost function at voxel for FBWM calculations.
            Refer to paper for additional information.

            Parameters
            ----------
            grid : array_like(dtype=float)
                Vector of values at which to compute cost. Usually 0 to 1
            BT : list of float
                List of unique B-value shells (eg. [1, 2, 6])
            GT : list of float
                List of gradient tables for each B-value shell
            b0 : float
                Averaged B0 signal in DWI
            IMG : list of float
                List of DWI signal for each B-value shell
            iDT : array_like(dtype=float)
                FBWM diffusion tensor
            iaDT : array_like(dtype=float)
                FBWM axonal diffusion tensor
            zeta : float
                Zeta value
            shB : list of complex
                SH basis sets for each B-value shell
            Pl0 : array_like(dtype=float)
                Legendre polynomail
            g2l_fa_R_b : array_like(dtype=complex)
                Information not provided
            clm : array_like(dtype=complex)
                fODF SH coefficients

            Returns
            -------
            cost_fn : array_like(dype=float)
                Cost values for input grid
            """

            if grid.ndim > 1:
                raise Exception("Grid needs to be a flattened 1D vector")
            ndir = [len(x) for x in GT]
            cost_fn = np.zeros_like(grid)
            with np.errstate(all="ignore"):
                for idx, awf in np.ndenumerate(grid):
                    for b in range(0, len(BT)):
                        if IMG[b].shape[0] != shB[b].shape[0]:
                            raise ValueError(
                            f"FBWM shell mismatch at shell {b}: "
                            f"signal has shape {IMG[b].shape}, "
                            f"basis has shape {shB[b].shape}"
                        )
                        Se = (
                            b0
                            * np.exp(
                                (-BT[b] * (1 - awf) ** -1)
                                * np.diag((GT[b].dot((iDT - (awf**3 * zeta**-2) * iaDT).dot(GT[b].T))))
                            )
                        ) * (1 - awf)  # Eq. 3 FBWM paper
                        Sa = (2 * np.pi * b0 * zeta * np.sqrt(np.pi / BT[b])) * (
                            shB[b].dot((Pl0 * g2l_fa_R_b[b, idx, :][0] * clm))
                        )  # Eq. 4 FBM paper
                        cost_fn[idx] = cost_fn[idx] + ndir[b] ** -1 * np.sum((IMG[b] - Se.real - Sa.real) ** 2)
                    cost_fn[idx] = b0**-1 * np.sqrt(len(BT) ** -1 * cost_fn[idx])  # Eq. 21 FBWM paper
            return cost_fn

        def fbi_helper(
            dwi,
            b0,
            B_pinv,
            B_shell,
            H,
            Pl0,
            gl,
            rectify=True,
            fbwm_SH1=None,
            fbwm_SH2=None,
            fbwm_B1=None,
            fbwm_B2=None,
            fbwm_dt=None,
            fbwm_degs=None,
            sh_area=None,
        ):
            """Computes FBI calculations for a given voxel. This function
            will perform FBWM only if all optional FBWM parameters are
            parsed.
            """
            fbwm = False
            if not (
                (fbwm_SH1 is None)
                or (fbwm_SH2 is None)
                or (fbwm_B1 is None)
                or (fbwm_B2 is None)
                or (fbwm_dt is None)
            ):
                fbwm = True

            if sh_area is None:
                rectify = False

            # ------------------------------------------------------------------
            # Safe input handling
            # ------------------------------------------------------------------
            dwi = np.asarray(dwi, dtype=np.float64).reshape(-1)
            b0 = float(np.asarray(b0).item())

            dwi = np.nan_to_num(dwi, nan=minZero, posinf=minZero, neginf=minZero)
            dwi[dwi < minZero] = minZero

            # ------------------------------------------------------------------
            # Compute SH coefficients safely
            # ------------------------------------------------------------------
            if (not np.isfinite(b0)) or (b0 <= minZero):
                alm = np.zeros(B_pinv.shape[0], dtype=np.complex128)
            else:
                with np.errstate(all="ignore"):
                    sig = dwi / b0
                sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

                with np.errstate(all="ignore"):
                    alm = np.dot(B_pinv, sig)

            alm = np.nan_to_num(alm, nan=0.0, posinf=0.0, neginf=0.0)

            # These are complex SH coefficients of the DWI signal
            a00 = alm[0].real

            with np.errstate(all="ignore"):
                clm = alm * gl[0] * highprecisionpower(
                    np.sqrt(4 * np.pi) * alm[0] * Pl0 * gl,
                    -1,
                )

            clm = np.nan_to_num(clm, nan=0.0, posinf=0.0, neginf=0.0)

            # ------------------------------------------------------------------
            # ODF reconstruction / optional rectification
            # ------------------------------------------------------------------
            with np.errstate(all="ignore"):
                ODF = np.matmul(H, clm)

            ODF = np.nan_to_num(ODF, nan=0.0, posinf=0.0, neginf=0.0)

            if rectify:
                ODF = __fbi_rectify(ODF.real, sh_area, iter=100)
                # Re-expand the rectified fODF into SH's
                with np.errstate(all="ignore"):
                    clm = np.matmul(sh_area * ODF, np.conj(H))
                clm = np.nan_to_num(clm, nan=0.0, posinf=0.0, neginf=0.0)

            # ------------------------------------------------------------------
            # Normalize clm safely
            # ------------------------------------------------------------------
            if (not np.isfinite(clm[0])) or (np.abs(clm[0]) < minZero):
                clm = np.zeros_like(clm, dtype=np.complex128)
                clm[0] = 1 / np.sqrt(4 * np.pi)
            else:
                clm = (clm / clm[0]) * (1 / np.sqrt(4 * np.pi))

            clm = np.nan_to_num(clm, nan=0.0, posinf=0.0, neginf=0.0)

            # ------------------------------------------------------------------
            # zeta and FAA calculations
            # NOTE: zeta is not affected by rectification, only FAA
            # ------------------------------------------------------------------
            zeta = a00 * np.sqrt(self.maxBval()) / np.pi

            with np.errstate(all="ignore"):
                faa = np.sqrt(
                    3 * np.sum(np.abs(clm[1:6] ** 2))
                    / (5 * np.abs(clm[0]) ** 2 + 2 * np.sum(np.abs(clm[1:6] ** 2)))
                )

            if not np.isfinite(faa):
                faa = 0.0

            # ------------------------------------------------------------------
            # BEGIN: construct axonal DT (aDT)
            # ------------------------------------------------------------------
            c00 = clm[0]
            c2_2 = clm[1]
            c2_1 = clm[2]
            c20 = clm[3]
            c21 = clm[4]
            c22 = clm[5]

            A11 = (np.sqrt(30) / 3) * c00 - (np.sqrt(6) / 3) * c20 + c22 + c2_2
            A22 = (np.sqrt(30) / 3) * c00 - (np.sqrt(6) / 3) * c20 - c22 - c2_2
            A33 = (np.sqrt(30) / 3) * c00 + (2 * np.sqrt(6) / 3) * c20
            A12 = 1j * (c22 - c2_2)
            A13 = -c21 + c2_1
            A23 = 1j * (-c21 - c2_1)

            aDT = np.array([A11, A12, A13, A12, A22, A23, A13, A23, A33]).real
            aDT = 1 / (c00 * np.sqrt(30)) * aDT
            iaDT = np.reshape(aDT, (3, 3)).real

            # ------------------------------------------------------------------
            # FBWM branch
            # ------------------------------------------------------------------
            if fbwm:
                bval = np.rint(self.grad[:, -1])
                BT = np.unique(bval)
                BT = BT[BT != 0]  # Exclude B0s
                GT = [self.grad[bval == x, 0:3] for x in BT]

                f_grid = np.linspace(0, 1, 100)
                int_grid = np.linspace(0, 99, 100, dtype=int)
                awf_grid = np.linspace(0, 1, 100)

                shB = [fbwm_SH1, fbwm_SH2, B_shell]
                IMG = [fbwm_B1, fbwm_B2, dwi]

                # BEGIN: DT construction
                iDT = np.array(
                    [
                        fbwm_dt[0],
                        fbwm_dt[3],
                        fbwm_dt[4],
                        fbwm_dt[3],
                        fbwm_dt[1],
                        fbwm_dt[5],
                        fbwm_dt[4],
                        fbwm_dt[5],
                        fbwm_dt[2],
                    ]
                )
                iDT = np.reshape(iDT, (3, 3))
                # END: DT construction

                g2l_fa_R = np.zeros((len(Pl0), f_grid.shape[0]), order="F")
                g2l_fa_R_b = np.zeros((len(BT), f_grid.shape[0], len(Pl0)), order="F")
                g2l_fa_R_large = np.zeros((len(Pl0), f_grid.shape[0]), order="F")

                for b in range(0, len(BT)):
                    idx_hyper = BT[b] * np.power(f_grid, 2) * np.power(zeta, -2) < 20
                    idx_Y = 0

                    for l_num in fbwm_degs[::2]:
                        hypergeom_opt = np.sum(
                            (
                                gamma((l_num + 1) / 2 + int_grid)
                                * gamma(l_num + (3 / 2))
                                * (
                                    (-BT[b] * f_grid[idx_hyper] ** 2 * zeta**-2)
                                    * np.ones((1, len(f_grid[idx_hyper])))
                                ).T
                                ** int_grid
                                / (
                                    factorial(int_grid)
                                    * gamma(l_num + (3 / 2) + int_grid)
                                    * gamma((l_num + 1) / 2)
                                )
                            ),
                            1,
                        ) * np.ones((1, len(f_grid[idx_hyper])))

                        g2l_fa_R[idx_Y : idx_Y + (2 * l_num + 1), np.squeeze(idx_hyper)] = np.tile(
                            (
                                factorial(l_num // 2)
                                * (BT[b] * f_grid[idx_hyper] ** 2 * zeta**-2) ** ((l_num + 1) / 2)
                                / gamma(l_num + (3 / 2))
                                * hypergeom_opt
                            ),
                            (2 * l_num + 1, 1),
                        )

                        idx_Y = idx_Y + (2 * l_num + 1)

                    g2l_fa_R_b[b, np.squeeze(idx_hyper), :] = g2l_fa_R[:, np.squeeze(idx_hyper)].T

                    idx_Y = 0
                    for l_num in fbwm_degs[::2]:
                        g2l_fa_R_large[idx_Y : idx_Y + (2 * l_num + 1), np.squeeze(~idx_hyper)] = np.tile(
                            np.exp(-l_num // 2 * (l_num + 1) / (2 * BT[b] * (f_grid[~idx_hyper] ** 2 * zeta**-2))),
                            (2 * l_num + 1, 1),
                        )
                        idx_Y = idx_Y + (2 * l_num + 1)

                    g2l_fa_R_b[b, np.squeeze(~idx_hyper), :] = g2l_fa_R_large[:, np.squeeze(~idx_hyper)].T

                cost_fn = __costCalculator(
                    awf_grid,
                    BT,
                    GT,
                    b0,
                    IMG,
                    iDT,
                    iaDT,
                    zeta,
                    shB,
                    Pl0,
                    g2l_fa_R_b,
                    clm,
                )

                min_cost_fn_idx = np.argsort(cost_fn, axis=0)
                min_cost_fn = np.take_along_axis(cost_fn, min_cost_fn_idx, axis=0)
                min_awf = awf_grid[min_cost_fn_idx[0]]

                De = (iDT - (min_awf**3 * zeta**-2) * iaDT) / (1 - min_awf)
                Da = min_awf**2 / zeta**2

                iDe = De
                iDe[np.isnan(iDe)] = minZero
                iDe[np.isinf(iDe)] = minZero

                L, V = np.linalg.eig(iDe)
                L = np.sort(L)[::-1]

                N = 1
                if L[0] < 0 or L[1] < 0 or L[2] < 0:
                    while L[0] < 0 or L[1] < 0 or L[2] < 0:
                        N = N + 1
                        if N < 100:
                            min_awf = awf_grid[min_cost_fn_idx[N]]
                        else:
                            min_awf = 0
                            break

                        # NOTE: keep your original zeta^-2 formulation here
                        De = (iDT - (min_awf**3 * zeta**-2) * iaDT) / (1 - min_awf)
                        Da = min_awf**2 / zeta**2

                        iDe = De
                        iDe[np.isnan(iDe)] = minZero
                        iDe[np.isinf(iDe)] = minZero
                        L, V = np.linalg.eig(iDe)
                        L = np.sort(L)[::-1]

                with np.errstate(invalid="ignore"):
                    De_ax = L[0]
                    De_rad = (L[1] + L[2]) / 2
                    De_fa = np.sqrt(
                        ((L[0] - L[1]) ** 2 + (L[0] - L[2]) ** 2 + (L[1] - L[2]) ** 2)
                        / (2 * np.sum(L**2))
                    )
                    De_mean = (1 / 3) * (2 * De_rad + De_ax)
                    min_cost = min_cost_fn[0]
            else:
                min_awf = None
                Da = None
                De_mean = None
                De_ax = None
                De_rad = None
                De_fa = None
                min_cost = None
                min_cost_fn = None

            return (
                zeta,
                faa,
                clm,
                min_awf,
                Da,
                De_mean,
                De_ax,
                De_rad,
                De_fa,
                min_cost,
                min_cost_fn,
            )
        # def fbi_helper(
        #     dwi,
        #     b0,
        #     B,
        #     H,
        #     Pl0,
        #     gl,
        #     rectify=True,
        #     fbwm_SH1=None,
        #     fbwm_SH2=None,
        #     fbwm_B1=None,
        #     fbwm_B2=None,
        #     fbwm_dt=None,
        #     fbwm_degs=None,
        #     sh_area=None,
        # ):
        #     """Computes FBI calculations for a given voxel. This function
        #     will perform FBWM only if all optional FBWM parameters are
        #     parsed.

        #     Parameters
        #     ----------
        #     dwi: array_like(dtype=float)
        #         Signal across DWI at a given voxel.
        #     b0: float
        #         Averaged B0 signal at a given voxel.
        #     B: array_like(dtype=complex)
        #         FBI spherical harmonic expansion.
        #     H: array_like(dtype=complex)
        #         ODF from spherical harmonic expansion.
        #     Pl0: array_like(dtype=float)
        #         Legendre polynomail.
        #     gl: array_like(dtype=float)
        #         Correction factor.
        #     rectify: bool; optional
        #         Specify whether to perform fODF rectification
        #         (Default: True).
        #     fbwm_SH1: array_like(dtype=complex); optional
        #         DKI spherical harmonic expansion for B1000.
        #     fbwm_SH2: array_like(dtype=complex); optional
        #         DKI spherical harmonic expansion for B2000.
        #     fbwm_degs: array_like(dtype=int); optional
        #         Harmonics used in DKI spherical harmonic expansion.
        #     fbwm_B1: array_like(dtype=float); optional
        #         Signal across DWI at B1000 at a given voxel.
        #     fbwm_B2: array_like(dtype=float); optional
        #         Signal across DWI at B2000 at a given voxel.
        #     fbwm_dt: array_like(dtype=float); optional
        #         Diffusion tensor at a given voxel.
        #     fbwm_degs: array_like(dtype=int)
        #         Harmonics used in expansion of FBWM shperical
        #         harmonics.
        #     sh_area: array_list(dtype=float)
        #         Area of spherical sampling.

        #     Returns
        #     -------
        #     zeta: float
        #         Zeta parameter
        #     faa: float
        #         Intra-axonal fractional anisotropy.
        #     clm: float
        #         Spherical harmonic coefficients.
        #     min_awf : float
        #         Axonal water fraction.
        #     Da: float
        #         Intrinsic intra-axonal diffusivity.
        #     De_mean: float
        #         Mean extra-axonal diffusion.
        #     De_ax: float
        #         Axial extra-axonal diffusion.
        #     De_rad: float
        #         Radial extra-axonal diffusion.
        #     De_fa: float
        #         Extra-axonal FA.
        #     min_cost: float
        #         Minimum cost of the cost function (first index of
        #         min_cost_fn).
        #     min_cost_fn: array_like(dtype=float)
        #         Cost function vector.
        #     """
        #     fbwm = False
        #     if not (
        #         (fbwm_SH1 is None) or (fbwm_SH2 is None) or (fbwm_B1 is None) or (fbwm_B2 is None) or (fbwm_dt is None)
        #     ):
        #         fbwm = True
        #     if sh_area is None:
        #         rectify = False
        #     # For references to alm and clm see FBI papers, they (alm
        #     # and clm) are defined in all of them
        #     alm = np.dot(np.linalg.pinv(B), (dwi / b0))  # DWI signal SH coefficients (these are complex)
        #     alm[np.isnan(alm)] = 0
        #     a00 = alm[0].real  # the imaginary part is on the order of 10^-18 (this is for zeta)
        #     clm = (
        #         alm * gl[0] * highprecisionpower(np.sqrt(4 * np.pi) * alm[0] * Pl0 * gl, -1)
        #     )  # fODF SH coefficients (these are complex)
        #     # need to figure out how to do peak detection (on this variable and
        #     # then read out odf structures like in MATLAB code) only the real
        #     # part would be read out but that would need to be done later on
        #     # after the rectification process below
        #     ODF = np.matmul(H, clm)
        #     if rectify:
        #         ODF = __fbi_rectify(ODF.real, sh_area, iter=100)
        #         # Re-expand the rectified fODF into SH's
        #         clm = np.matmul(sh_area * ODF, np.conj(H))
        #     clm = (clm / clm[0]) * (1 / np.sqrt(4 * np.pi))  # normalize clm
        #     # zeta and FAA calculations
        #     # NOTE: zeta is not affected by the rectification, only FAA
        #     zeta = a00 * np.sqrt(self.maxBval()) / np.pi
        #     faa = np.sqrt(
        #         3 * np.sum(np.abs(clm[1:6] ** 2)) / (5 * np.abs(clm[0]) ** 2 + 2 * np.sum(np.abs(clm[1:6] ** 2)))
        #     )
        #     # BEGIN: construct axonal DT (aDT)
        #     c00 = clm[0]
        #     c2_2 = clm[1]
        #     c2_1 = clm[2]
        #     c20 = clm[3]
        #     c21 = clm[4]
        #     c22 = clm[5]
        #     A11 = (np.sqrt(30) / 3) * c00 - (np.sqrt(6) / 3) * c20 + c22 + c2_2
        #     A22 = (np.sqrt(30) / 3) * c00 - (np.sqrt(6) / 3) * c20 - c22 - c2_2
        #     A33 = (np.sqrt(30) / 3) * c00 + (2 * np.sqrt(6) / 3) * c20
        #     A12 = 1j * (c22 - c2_2)
        #     A13 = -c21 + c2_1
        #     A23 = 1j * (-c21 - c2_1)
        #     aDT = np.array([A11, A12, A13, A12, A22, A23, A13, A23, A33]).real
        #     aDT = 1 / (c00 * np.sqrt(30)) * aDT
        #     iaDT = np.reshape(aDT, (3, 3)).real
        #     if fbwm:
        #         bval = np.rint(self.grad[:, -1])
        #         BT = np.unique(bval)
        #         BT = BT[BT != 0]  # Exclude B0s
        #         GT = [self.grad[bval == x, 0:3] for x in BT]
        #         f_grid = np.linspace(0, 1, 100)  # define AWF grid (100 pts evenly spaced between 0 (min) and 1 (max))
        #         int_grid = np.linspace(0, 99, 100, dtype=int)  # define grid points to iterate over (100 of them)
        #         awf_grid = np.linspace(0, 1, 100)  # another AWF grid
        #         # This holds the SH basis sets for each b-value shell
        #         shB = [
        #             fbwm_SH1,
        #             fbwm_SH2,
        #             B,
        #         ]  # list object: to access, shB[0] = B1 (for example)
        #         # This hold all DWI volumes for each b-vlaue shell
        #         IMG = [fbwm_B1, fbwm_B2, dwi]  # list object: to access
        #         # BEGIN: DT construction
        #         iDT = np.array(
        #             [
        #                 fbwm_dt[0],
        #                 fbwm_dt[3],
        #                 fbwm_dt[4],
        #                 fbwm_dt[3],
        #                 fbwm_dt[1],
        #                 fbwm_dt[5],
        #                 fbwm_dt[4],
        #                 fbwm_dt[5],
        #                 fbwm_dt[2],
        #             ]
        #         )
        #         iDT = np.reshape(iDT, (3, 3))
        #         # END: DT construction
        #         # initialze correction factor elements that will be looped over and filled accordingly...
        #         g2l_fa_R = np.zeros((len(Pl0), f_grid.shape[0]), order="F")
        #         g2l_fa_R_b = np.zeros((len(BT), f_grid.shape[0], len(Pl0)), order="F")
        #         g2l_fa_R_large = np.zeros((len(Pl0), f_grid.shape[0]), order="F")
        #         # BEGIN: cost function
        #         # Not many comments here, See McKinnon 2018 FBWm paper for details
        #         for b in range(0, len(BT)):
        #             idx_hyper = (
        #                 BT[b] * np.power(f_grid, 2) * np.power(zeta, -2) < 20
        #             )  # when should hypergeometric function be implemented? When b*D is small
        #             idx_Y = 0
        #             for l_num in degs[::2]:
        #                 hypergeom_opt = np.sum(
        #                     (
        #                         gamma((l_num + 1) / 2 + int_grid)
        #                         * gamma(l_num + (3 / 2))
        #                         * (
        #                             (-BT[b] * f_grid[idx_hyper] ** 2 * zeta**-2) * np.ones((1, len(f_grid[idx_hyper])))
        #                         ).T
        #                         ** int_grid
        #                         / (factorial(int_grid) * gamma(l_num + (3 / 2) + int_grid) * gamma((l_num + 1) / 2))
        #                     ),
        #                     1,
        #                 ) * np.ones((1, len(f_grid[idx_hyper])))
        #                 g2l_fa_R[idx_Y : idx_Y + (2 * l_num + 1), np.squeeze(idx_hyper)] = np.tile(
        #                     (
        #                         factorial(l_num // 2)
        #                         * (BT[b] * f_grid[idx_hyper] ** 2 * zeta**-2) ** ((l_num + 1) / 2)
        #                         / gamma(l_num + (3 / 2))
        #                         * hypergeom_opt
        #                     ),
        #                     (2 * l_num + 1),
        #                     1,
        #                 )  # Eq. 9 FBWM paper
        #                 idx_Y = idx_Y + (2 * l_num + 1)
        #             g2l_fa_R_b[b, np.squeeze(idx_hyper), :] = g2l_fa_R[:, np.squeeze(idx_hyper)].T
        #             idx_Y = 0
        #             for l_num in degs[::2]:
        #                 g2l_fa_R_large[idx_Y : idx_Y + (2 * l_num + 1), np.squeeze(~idx_hyper)] = np.tile(
        #                     (np.exp(-l_num // 2 * (l_num + 1) / (2 * BT[b] * (f_grid[~idx_hyper] ** 2 * zeta**-2)))),
        #                     ((2 * l_num + 1),
        #                     1,
        #                 ))  # Eq. 20 FBI paper
        #                 idx_Y = idx_Y + (2 * l_num + 1)
        #             g2l_fa_R_b[b, np.squeeze(~idx_hyper), :] = g2l_fa_R_large[:, np.squeeze(~idx_hyper)].T
        #         cost_fn = __costCalculator(
        #             awf_grid,
        #             BT,
        #             GT,
        #             b0,
        #             IMG,
        #             iDT,
        #             iaDT,
        #             zeta,
        #             shB,
        #             Pl0,
        #             g2l_fa_R_b,
        #             clm,
        #         )
        #         min_cost_fn_idx = np.argsort(cost_fn, axis=0)  # find the indexes of the sorted cost_fn values
        #         min_cost_fn = np.take_along_axis(cost_fn, min_cost_fn_idx, axis=0)  # sort those values
        #         min_awf = awf_grid[
        #             min_cost_fn_idx[0]
        #         ]  # grad the minimum AWF value based on the cost_fn sorting done immeidately prior to this...
        #         De = (iDT - (min_awf**3 * zeta**-2) * iaDT) / (1 - min_awf)
        #         Da = min_awf**2 / zeta**2
        #         iDe = De  # intermeidate De
        #         iDe[np.isnan(iDe)] = minZero
        #         iDe[np.isinf(iDe)] = minZero
        #         L, V = np.linalg.eig(iDe)  # L : eigVals and V: eigVecs
        #         L = np.sort(L)  # sort them (this is ascending)
        #         L = L[::-1]  # reverse the order so they are descending (high -> low)
        #         N = 1  # initialize counter
        #         if L[0] < 0 or L[1] < 0 or L[2] < 0:
        #             while L[0] < 0 or L[1] < 0 or L[2] < 0:  # find new AWF values if L's are < 0
        #                 N = N + 1
        #                 if N < 100:
        #                     min_awf = awf_grid[min_cost_fn_idx[N]]
        #                 else:
        #                     min_awf = 0
        #                     break
        #                 # update De here...
        #                 De = (iDT - (min_awf**3 * zeta**2) * iaDT) / (1 - min_awf)
        #                 Da = min_awf**2 / zeta**2  # recalculate Da too...
        #             # Now recalculate eigVals again with correct AWF values
        #             iDe = De
        #             iDe[np.isnan(iDe)] = minZero
        #             iDe[np.isinf(iDe)] = minZero
        #             L, V = np.linalg.eig(iDe)  # L : eigVals and V: eigVecs
        #             L = np.sort(L)  # again, ascending
        #             L = L[::-1]  # now, descending
        #         with np.errstate(invalid="ignore"):
        #             De_ax = L[0]  # Eq. 24 FBWM paper, axial extra-axonal diffusivity
        #             De_rad = (L[1] + L[2]) / 2  # radial De
        #             De_fa = np.sqrt(
        #                 ((L[0] - L[1]) ** 2 + (L[0] - L[2]) ** 2 + (L[1] - L[2]) ** 2) / (2 * np.sum(L**2))
        #             )  # extra-axonal FA
        #             De_mean = (1 / 3) * (2 * De_rad + De_ax)  # average De
        #             min_cost = min_cost_fn[0]
        #     else:
        #         min_awf = None
        #         Da = None
        #         De_mean = None
        #         De_ax = None
        #         De_rad = None
        #         De_fa = None
        #         min_cost = None
        #         min_cost_fn = None

        #     return (
        #         zeta,
        #         faa,
        #         clm,
        #         min_awf,
        #         Da,
        #         De_mean,
        #         De_ax,
        #         De_rad,
        #         De_fa,
        #         min_cost,
        #         min_cost_fn,
        #     )

        # --------------------FUNCTION SEPARATOR-----------------------

        if fbwm and not hasattr(self, "dt"):
            raise Exception(
                "Cannot compute FBWM parameters "
                "without running diffusion tensor fitting first. "
                "Please run DWI.fit(constraints) before running DWI.fbi()."
            )
        if l_max % 2 != 0:
            raise Exception("Please provide l_max as a postive and even integer")
        if l_max > self.optimal_lmax():
            print(
                "[WARNING]: l_max value provided ({}) is "
                "more than that supported by DWI ({}). Reverting "
                "to l_max = {}".format(l_max, self.optimal_lmax(), self.optimal_lmax())
            )
            l_max = self.optimal_lmax()
        l_max = int(np.asarray(l_max).item())
        img = self.img
        np.unique(self.grad[:, -1])
        self.optimal_lmax()
        b0 = np.nanmean(img[:, :, :, self.idxb0()], axis=3)
        b0 = np.nan_to_num(b0, nan=0, posinf=0, neginf=0)
        # Vectorize images
        b0 = vectorize(b0, self.mask)
        img = vectorize(img, self.mask)
        # Create shperical harmonic (SH) base set
        degs = np.arange(l_max + 1, dtype=int)
        l_num = 2 * degs + 1  # how many per degree (evens only)
        harmonics = []
        sh_end = 0  # initialize the SH set for indexing
        for _, phase in enumerate(l_num[::2]):
            sh_start = sh_end + phase - 1
            sh_end = sh_start + phase - 1
            harmonics.extend(np.arange(sh_start, sh_end + 1))
        # Define the azimuthal (phi) and polar(theta) angles for our
        # spherical expansion using the experimentally defined
        # gradients from the scanner
        phi = np.arccos(self.grad[self.idxfbi(), 2])
        theta = np.arctan2(self.grad[self.idxfbi(), 1], self.grad[self.idxfbi(), 0])
        # gradients for resampling from distribution
        (
            spherical_grid,
            idx,
            idx8,
            AREA,
            faces,
            separation_angle,
        ) = sphericalsampling.odfgrid(res)
        S1 = spherical_grid[:, 0]  # phi
        S2 = spherical_grid[:, 1]  # theta
        # B = shbasis(degs, phi, theta)
        # H = shbasis(degs, S1, S2)
        B = odf.shbasis(degs, phi, theta, method="scipy")
        B = B[:, harmonics]
        H = odf.shbasis(degs, S1, S2, method="scipy")
        H = H[:, harmonics]

        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        B_shell = B.copy()

        with np.errstate(all="ignore"):
            B_pinv = np.linalg.pinv(B, rcond=1e-10)
        B_pinv = np.nan_to_num(B_pinv, nan=0.0, posinf=0.0, neginf=0.0)

        idx_Y = 0
        Pl0 = np.zeros((len(harmonics), 1), order="F")  # need Legendre polynomial Pl0
        gl = np.zeros(
            (len(harmonics), 1), order="F"
        )  # calculate correction factor (see original FBI paper, Jensen 2016)
        for l_num in degs[::2]:
            Pl0[idx_Y : idx_Y + (2 * l_num + 1), :] = (
                (np.power(-1, l_num // 2) * math.factorial(l_num))
                / (np.power(4, l_num // 2) * np.power(math.factorial(l_num // 2), 2))
                * np.ones((2 * l_num + 1, 1))
            )
            gl[idx_Y : idx_Y + (2 * l_num + 1), :] = (
                (math.factorial(l_num // 2) * np.power(self.maxBval() * th.__d0__, (l_num + 1) / 2))
                / gamma(l_num + 3 / 2)
                * hyp1f1((l_num + 1) / 2, l_num + 3 / 2, -self.maxBval() * th.__d0__)
                * np.ones((2 * l_num + 1, 1))
            )
            idx_Y = idx_Y + (2 * l_num + 1)
        Pl0 = np.squeeze(Pl0)
        gl = np.squeeze(gl)
        inputs = tqdm(
            range(0, img.shape[1]),
            desc="FBI Fit",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="vox",
            ncols=tqdmWidth,
        )
        if fbwm:
            # Index gradients based on b1000 and b2000 shells
            bval = np.rint(self.grad[:, -1])
            phi1 = np.arccos(self.grad[bval == 1, 2])
            theta1 = np.arctan2(self.grad[bval == 1, 1], self.grad[bval == 1, 0])

            phi2 = np.arccos(self.grad[bval == 2, 2])
            theta2 = np.arctan2(self.grad[bval == 2, 1], self.grad[bval == 2, 0])
            # SH basis set for the two B-values in DKI
            fbwm_SH1 = odf.shbasis(degs, phi1, theta1, method="scipy")
            fbwm_SH2 = odf.shbasis(degs, phi2, theta2, method="scipy")
            fbwm_SH1 = fbwm_SH1[:, harmonics]
            fbwm_SH2 = fbwm_SH2[:, harmonics]
            dt, kt = self.tensorReorder("dki")
            dt = vectorize(dt, self.mask)

            (
                zeta,
                faa,
                fodf,
                min_awf,
                Da,
                De_mean,
                De_ax,
                De_rad,
                De_fa,
                min_cost,
                min_cost_fn,
            ) = zip(
                *Parallel(n_jobs=self.workers, prefer="processes")(
                    delayed(fbi_helper)(
                        dwi=img[self.idxfbi(), i],
                        b0=b0[i],
                        B_pinv=B_pinv,
                        B_shell=B_shell,
                        H=H,
                        Pl0=Pl0,
                        gl=gl,
                        rectify=rectify,
                        fbwm_SH1=fbwm_SH1,
                        fbwm_SH2=fbwm_SH2,
                        fbwm_B1=img[bval == 1, i],
                        fbwm_B2=img[bval == 2, i],
                        fbwm_dt=dt[:, i],
                        fbwm_degs=degs,
                        sh_area=AREA,
                    )
                    for i in inputs
                )
            )
        else:

            (
                zeta,
                faa,
                fodf,
                min_awf,
                Da,
                De_mean,
                De_ax,
                De_rad,
                De_fa,
                min_cost,
                min_cost_fn,
            ) = zip(
                *Parallel(n_jobs=self.workers, prefer="processes")(
                    delayed(fbi_helper)(
                        dwi=img[self.idxfbi(), i],
                        b0=b0[i],
                        B_pinv=B_pinv,
                        B_shell=B_shell,
                        H=H,
                        Pl0=Pl0,
                        gl=gl,
                        rectify=rectify,
                        sh_area=AREA,
                    )
                    for i in inputs
                )
            )
        fodf_mrtrix = odf.odf_conversion(np.array(fodf).T, target="tournier")
        zeta = vectorize(np.array(zeta), self.mask)
        faa = vectorize(np.array(faa), self.mask)
        fodf = vectorize(np.array(fodf).T, self.mask)
        fodf_mrtrix = vectorize(fodf_mrtrix, self.mask)
        awf = vectorize(np.array(min_awf), self.mask)
        Da = vectorize(np.array(Da), self.mask)
        De_mean = vectorize(np.array(De_mean), self.mask)
        De_ax = vectorize(np.array(De_ax), self.mask)
        De_rad = vectorize(np.array(De_rad), self.mask)
        De_fa = vectorize(np.array(De_fa), self.mask)
        min_cost = vectorize(np.array(min_cost), self.mask)
        min_cost_fn = vectorize(np.array(min_cost_fn).T, self.mask)
        return (
            zeta,
            faa,
            fodf,
            fodf_mrtrix,
            awf,
            Da,
            De_mean,
            De_ax,
            De_rad,
            De_fa,
            min_cost,
            min_cost_fn,
        )

    def extractWMTI(
        self,
    ) -> Tuple[
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
    ]:
        """Returns white matter tract integrity (WMTI) parameters. Warning:
        this can only be run after fitting and DWI.extractDTI().

        Returns
        -------
        awf: ndarray(dtype=float)
            Axonal Water Fraction
        eas_ad: ndarray(dtype=float)
            Extra-axonal space Axial Diffusivity
        eas_rd: ndarray(dtype=float)
            Extra-axonal Space Radial Diffusivity
        eas_md: ndarray(dtype=float)
            Extra-axonal Space Mean Diffusivity
        eas_tort: ndarray(dtype=float)
            Extra-axonal Space Tortuosity
        ias_ad: ndarray(dtype=float)
            Intra-axonal Space Axial Diffusivity
        ias_rd: ndarray(dtype=float)
            Intra-axonal Space Radial Diffusivity
        ias_da: ndarray(dtype=float)
            Intra-axonal Space Intrinsic Diffusivity
        ias_tort: ndarray(dtype=float)
            Intra-axonal Space Tortuosity
        """

        def wmtihelper(dt, dir, adc, akc, awf, adc2dt):
            # Avoid complex output. However,
            # negative AKC might be taken care of by applying constraints
            with np.errstate(invalid="ignore"):
                akc[akc < minZero] = minZero
            try:
                # Eigenvalue decomposition of De(extra-axonal)
                De = np.multiply(adc, 1 + np.sqrt((np.multiply(akc, awf) / (3 * (1 - awf)))))
                dt_e = np.matmul(adc2dt, De)
                DTe = dt_e[[0, 1, 2, 1, 3, 4, 2, 4, 5]]
                DTe = np.reshape(DTe, (3, 3), order="F")
                eigval = sla.eigh(DTe, eigvals_only=True)
                eigval = np.sort(eigval)[::-1]
                eas_ad = eigval[0]
                eas_rd = 0.5 * (eigval[1] + eigval[2])
                try:
                    eas_tort = eas_ad / eas_rd
                except:  # noqa: E722
                    eas_tort = minZero
            except:  # noqa: E722
                eas_ad = minZero
                eas_rd = minZero
                eas_tort = minZero
            try:
                # Eigenvalue decomposition of Da (intra-axonal)
                Di = np.multiply(adc, 1 - np.sqrt((np.multiply(akc, (1 - awf)) / (3 * awf))))
                dt_i = np.matmul(adc2dt, Di)
                DTi = dt_i[[0, 1, 2, 1, 3, 4, 2, 4, 5]]
                DTi = np.reshape(DTi, (3, 3), order="F")
                eigval = sla.eigh(DTi, eigvals_only=True)
                eigval = np.sort(eigval)[::-1]
                ias_da = np.sum(eigval)
                np.seterr(invalid="raise")
            except:  # noqa: E722
                ias_da = minZero
            return eas_ad, eas_rd, eas_tort, ias_da

        dir = dwidirs.dirs10000
        nvox = self.dt.shape[1]
        N = dir.shape[0]
        nblocks = 10
        maxk = np.zeros((nvox, nblocks)).astype(float)
        inputs = tqdm(
            range(nblocks),
            desc="Extracting AWF",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="iter",
            ncols=tqdmWidth,
        )
        for i in inputs:
            maxk = np.stack(self.kurtosisCoeff(self.dt, dir[int(N / nblocks * i) : int(N / nblocks * (i + 1))])).astype(
                float
            )
            maxk = np.nanmax(maxk, axis=0)
        awf = np.divide(maxk, (maxk + 3)).astype(float)
        # Changes voxels less than minZero, nans and infs to minZero
        truncateIdx = np.logical_or(np.logical_or(np.isnan(awf), np.isinf(awf)), (awf < minZero))
        awf[truncateIdx] = minZero
        dirs = dwidirs.dirs30
        adc = self.diffusionCoeff(self.dt[:6], dirs)
        akc = self.kurtosisCoeff(self.dt, dirs)
        (dcnt, dind) = self.createTensorOrder(2)
        adc2dt = np.linalg.pinv(np.matmul((dirs[:, dind[:, 0]] * dirs[:, dind[:, 1]]), np.diag(dcnt)))
        eas_ad = np.zeros(nvox)
        eas_rd = np.zeros(nvox)
        eas_md = np.zeros(nvox)
        eas_tort = np.zeros(nvox)
        ias_ad = np.zeros(nvox)
        ias_rd = np.zeros(nvox)
        ias_da = np.zeros(nvox)
        ias_tort = np.zeros(nvox)
        inputs = tqdm(
            range(nvox),
            desc="Extracting EAS and IAS",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="vox",
            ncols=tqdmWidth,
        )
        eas_ad, eas_rd, eas_tort, ias_da = zip(
            *Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(wmtihelper)(self.dt[:, i], dirs, adc[:, i], akc[:, i], awf[i], adc2dt) for i in inputs
            )
        )
        awf = vectorize(awf, self.mask)
        eas_ad = vectorize(np.array(eas_ad), self.mask)
        eas_rd = vectorize(np.array(eas_rd), self.mask)
        eas_md = vectorize(np.array(eas_md), self.mask)
        eas_tort = vectorize(np.array(eas_tort), self.mask)
        ias_ad = vectorize(np.array(ias_ad), self.mask)
        ias_rd = vectorize(np.array(ias_rd), self.mask)
        ias_da = vectorize(np.array(ias_da), self.mask)
        ias_tort = vectorize(np.array(ias_tort), self.mask)
        return awf, eas_ad, eas_rd, eas_tort, ias_da

    def multiplyMask(self, img: np.ndarray[float]) -> np.ndarray[float]:
        """Multiplies a 3D image by the brain mask.

        Parameters
        ----------
        img: ndarray(dtype=float)
            3D image to be multiplied.

        Returns
        -------
        ndarray(dtype=float)
            Multiplied image.
        """
        # Returns an image multiplied by the brain mask to remove all
        # values outside the brain
        return np.multiply(self.mask.astype(bool), img)

    def akcoutliers(self, iter: int = 10) -> np.ndarray[bool]:
        """Uses 100,000 direction in chunks of 10 to iteratively find
        outliers. Returns a mask of locations where said violations
        occur. Multiprocessing is disabled because this is a
        memory-intensive task.
        To be run only after tensor fitting.

        Parameters
        ----------
        iter: int, optional
            Number of iterations to perform out of 10. Reduce this
            number if your computer does not have sufficient RAM.
            (Default: 10).

        Returns
        -------
        akc_out: ndarray(dtype=bool)
            3D map containing outliers where AKC falls fails the
            inequality test -2 < AKC < 10.

        Examples
        --------
        akc_out = dwi.akoutliers(), where dwi is the DWI class object.
        """
        dir = dwidirs.dirs10000
        nvox = self.dt.shape[1]
        akc_out = np.zeros(nvox, dtype=bool)
        N = dir.shape[0]
        nblocks = 10
        if iter > nblocks:
            print("Entered iteration value exceeds 10...resetting to 10")
            iter = 10
        inputs = tqdm(
            range(iter),
            desc="AKC Outlier Detection",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="blk",
            ncols=tqdmWidth,
        )
        for i in inputs:
            akc = self.kurtosisCoeff(self.dt, dir[int(N / nblocks * i) : int(N / nblocks * (i + 1))])
            akc_out[np.where(np.any(np.logical_or(akc < -2, akc > 10), axis=0))] = True
            akc_out.astype("bool")
        return vectorize(akc_out, self.mask)

    def akccorrect(self, akc_out: np.ndarray[bool], window: int = 3, connectivity: str = "face") -> Self:
        """Applies AKC outlier map to DT to replace outliers with a
        moving median. Run this only after tensor fitting and akc
        outlier detection.

        Parameters
        ----------
        akc_out : ndarray(dtype=bool)
            3D map containing outliers from DWI.akcoutliers
        window : int, optional
            Width of square matrix filter (Default: 5)
        connectivity : str, {'face', 'all'}, optional
            Specifies what kind of connected-component connectivity to
            use for median determination

        Examples
        --------
        dwi.akccorrect(akc_out), where dwi is the DWI class object
        """
        # Get box filter properties
        centralIdx = np.median(range(window))
        d2move = int(np.abs(window - (centralIdx + 1)))  # Add 1 to
        # central idx because first index starts with zero
        # Vectorize and Pad
        dt = np.pad(
            vectorize(self.dt, self.mask),
            ((d2move, d2move), (d2move, d2move), (d2move, d2move), (0, 0)),
            "constant",
            constant_values=np.nan,
        )
        akc_out = np.pad(akc_out, d2move, "constant", constant_values=False)
        violIdx = np.array(np.where(akc_out))  # Locate coordinates of violations
        nvox = violIdx.shape[1]
        for i in tqdm(
            range(dt.shape[-1]),
            desc="AKC Correction",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="tensor",
            ncols=tqdmWidth,
        ):
            for j in range(nvox):
                # Index beginning and ending of patch
                Ib = violIdx[0, j] - d2move
                Ie = violIdx[0, j] + d2move + 1
                Jb = violIdx[1, j] - d2move
                Je = violIdx[1, j] + d2move + 1
                Kb = violIdx[2, j] - d2move
                Ke = violIdx[2, j] + d2move + 1

                if connectivity == "all":
                    patchViol = np.delete(
                        np.ravel(akc_out[Ib:Ie, Jb:Je, Kb:Ke]),
                        np.median(range(np.power(window, 3))),
                    )  # Remove
                    # centroid element
                    patchImg = np.delete(
                        np.ravel(dt[Ib:Ie, Jb:Je, Kb:Ke, i]),
                        np.median(range(np.power(window, 3))),
                    )  # Remove
                    # centroid element
                    connLimit = np.power(window, 3) - 1
                elif connectivity == "face":
                    patchViol = np.delete(akc_out[Ib:Ie, violIdx[1, j], violIdx[2, j]], d2move)
                    patchViol = np.hstack(
                        (
                            patchViol,
                            np.delete(akc_out[violIdx[0, j], Jb:Je, violIdx[2, j]], d2move),
                        )
                    )
                    patchViol = np.hstack(
                        (
                            patchViol,
                            np.delete(akc_out[violIdx[0, j], violIdx[1, j], Kb:Ke], d2move),
                        )
                    )
                    patchImg = np.delete(dt[Ib:Ie, violIdx[1, j], violIdx[2, j], i], d2move)
                    patchImg = np.hstack(
                        (
                            patchImg,
                            np.delete(dt[violIdx[0, j], Jb:Je, violIdx[2, j], i], d2move),
                        )
                    )
                    patchImg = np.hstack(
                        (
                            patchImg,
                            np.delete(dt[violIdx[0, j], violIdx[1, j], Kb:Ke, i], d2move),
                        )
                    )
                    if window == 3:
                        connLimit = 6
                    elif window == 5:
                        connLimit = 12
                    elif window == 7:
                        connLimit = 18
                    elif window == 9:
                        connLimit = 24
                else:
                    raise Exception(
                        'Connectivity choice "{}" is invalid. Please enter either "all" or "face".'.format(connectivity)
                    )
                nVoil = np.sum(patchViol)

                # Here a check is performed to compute the number of
                # violations in a patch. If all voxels are violations,
                # do nothing. Otherwise, exclude violation voxels from
                # the median calculation
                if nVoil == connLimit:
                    continue
                else:
                    dt[violIdx[0, j], violIdx[1, j], violIdx[2, j], i] = np.nanmedian(patchImg)
        # Remove padding
        dt = dt[d2move:-d2move, d2move:-d2move, d2move:-d2move, :]
        self.dt = vectorize(dt, self.mask)

    def irlls(
        self,
        excludeb0: bool = True,
        maxiter: int = 25,
        convcrit: float = 1e-3,
        mode: str = "DKI",
        leverage: float = 0.85,
        bounds: int = 3,
    ) -> Tuple[np.ndarray[bool], np.ndarray[float]]:
        """This functions performs outlier detection and robust parameter
        estimation for diffusion MRI using the iterative reweigthed
        linear least squares (IRLLS) approach.

        Parameters
        ----------
        exludeb0: bool, optional
            Exlude the b0 images when removing outliers (Default: True)
        maxiter: int, optional
            Maximum number of iterations in the iterative reweighting
            loop (Default: 25).
        convcrit: float, optional
            Fraction of L2-norm of estimated diffusion parameter
            vector that the L2-norm of different vector should get in
            order to reach convergence in the iterative reweighted
            loop (Default: 1e-3).
        mode: str, {'DKI', 'DTI'}, optional
            Specifies whether to use DTI or DKI model (Default: 'DKI')
        leverage: float, optional.
            Measurement ranging from 0 to 1 where a leverage above
            this threshold will not be excluded in estimation of DT
            after outlier (Default: 0.85).
        Bounds: int, optional
            Set the threshold of the number of standard deviation
            that are needed to exclude a measurement (Default: 3).

        Returns
        -------
        outliers: ndarray(dtype=bool)
            4D image same size as input DWI marking voxels
            that are outliers.
        dt: ndarray(dtype=float)
            IRLLS method of DT estimation.

        Examples
        --------
        outliers = dwi.irlls()
        """
        # if not excludeb0.dtype:
        #     assert('option: Excludeb0 should be set to True or False')

        if maxiter < 1 or maxiter > 200:
            assert "option: Maxiter should be set to a value between 1 and 200"
        if convcrit < 0 or convcrit > 1:
            assert "option: Maxiter should be set to a value between 1 and 200"
        if not (mode == "DKI" or mode == "DTI"):
            assert "Mode should be set to DKI or DTI"
        if leverage < 0 or leverage > 1:
            assert "option: Leverage should be set to a value between 0 and 1"
        if bounds < 1:
            assert "option: Bounds should be set to a value >= 1"
        exclude_idx = np.ones_like(self.grad[:, 3], dtype=bool)
        exclude_idx = self.idxdki()
        # Vectorize DWI
        dwi = vectorize(self.img[:, :, :, exclude_idx], self.mask)
        (ndwi, nvox) = dwi.shape
        b = np.array(self.grad[exclude_idx, 3])
        b = np.reshape(b, (len(b), 1))
        g = self.grad[exclude_idx, 0:3]
        # Apply Scaling
        scaling = False
        if np.sum(dwi < 1) / np.size(dwi) < 0.001:
            dwi[dwi < 1] = 1
        else:
            scaling = True
            if self.maxDKIBval() < 10:
                tmp = dwi[dwi < 0.05]
            else:
                tmp = dwi[dwi < 50]
            sc = np.median(tmp)
            dwi[dwi < sc / 1000] = sc / 1000
            dwi = dwi * 1000 / sc
        # Create B-matrix
        (dcnt, dind) = self.createTensorOrder(2)
        if mode == "DTI":
            bmat = np.hstack(
                (
                    np.ones((ndwi, 1)),
                    np.matmul(
                        (-np.tile(b, (1, 6)) * g[:, dind[:, 0]] * g[:, dind[:, 1]]),
                        np.diag(dcnt),
                    ),
                )
            )
        else:
            (wcnt, wind) = self.createTensorOrder(4)
            bmat = np.hstack(
                (
                    np.ones((ndwi, 1)),
                    np.matmul(
                        (-np.tile(b, (1, 6)) * g[:, dind[:, 0]] * g[:, dind[:, 1]]),
                        np.diag(dcnt),
                    ),
                    (1 / 6)
                    * np.matmul(
                        (
                            np.square(np.tile(b, (1, 15)))
                            * g[:, wind[:, 0]]
                            * g[:, wind[:, 1]]
                            * g[:, wind[:, 2]]
                            * g[:, wind[:, 3]]
                        ),
                        np.diag(wcnt),
                    ),
                )
            )
        nparam = bmat.shape[1]
        ndof = ndwi - nparam
        # Initialization
        b0_pos = np.zeros(b.shape, dtype=bool, order="F")
        if excludeb0:
            if self.maxDKIBval() < 10:
                b0_pos = b < 0.01
            else:
                b0_pos = b < 10
        reject = np.zeros(dwi.shape, dtype=bool, order="F")
        np.zeros((nvox, 1))
        dt = np.zeros((nparam, nvox))
        # Attempt basic noise estimation
        try:
            sigma
        except NameError:

            def estSigma(dwi, bmat):
                dwi = np.reshape(dwi, (len(dwi), 1))
                try:
                    dt_ = np.linalg.lstsq(bmat, np.log(dwi), rcond=None)[0]
                    # dt_ = np.linalg.solve(np.dot(bmat.T, bmat), np.dot(
                    #     bmat.T, np.log(dwi)))
                except:  # noqa: E722
                    dt_ = np.full((bmat.shape[1], 1), minZero)


                with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                    z = np.matmul(bmat, dt_)

                # Clamp exponent to a safe range
                z = np.clip(z, -50.0, 50.0)

                # w = highprecisionexp(np.matmul(bmat, dt_)).reshape((ndwi, 1))
                w = highprecisionexp(z).reshape((ndwi, 1))
                w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

                if np.random.rand() < 1e-5:
                    print("w stats:", np.min(w), np.max(w))

                try:
                    dt_ = np.linalg.lstsq((bmat * np.tile(w, (1, nparam))), (np.log(dwi) * w), rcond=None)[0]
                    # dt_ = np.linalg.solve(
                    #     np.dot((bmat * np.tile(w, (1, nparam))).T,
                    #         (bmat * np.tile(w, (1, nparam)))), \
                    #     np.dot((bmat * np.tile(w, (1, nparam))).T, (np.log(
                    #         dwi) * w)))
                except:  # noqa: E722
                    dt_ = np.full((bmat.shape[1], 1), minZero)
                e = np.log(dwi) - np.matmul(bmat, dt_)
                m = np.median(np.abs((e * w) - np.median(e * w)))
                try:
                    sigma_ = np.sqrt(ndwi / ndof) * 1.4826 * m
                except:  # noqa: E722
                    sigma_ = minZero
                return sigma_

            sigma_ = np.zeros((nvox, 1))
            inputs = tqdm(
                range(nvox),
                desc="IRLLS Noise Estimation",
                bar_format="{desc}: [{percentage:0.0f}%]",
                unit="vox",
                ncols=tqdmWidth,
            )
            sigma_ = Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(estSigma)(dwi[:, i], bmat) for i in inputs
            )
            sigma = np.median(sigma_)
            sigma = np.tile(sigma, (nvox, 1))
        if scaling:
            sigma = sigma * 1000 / sc

        def outlierHelper(dwi, bmat, sigma, b, b0_pos, maxiter=25, convcrit=1e-3, leverage=3, bounds=3):
            # Preliminary rough outlier check
            dwi_i = dwi.reshape((len(dwi), 1))
            dwi0 = np.median(dwi_i[b.reshape(-1) < 0.01])
            out = dwi_i > (dwi0 + 3 * sigma)

            if np.sum(~out[b.reshape(-1) > 0.01]) < (bmat.shape[1] - 1):
                out = np.zeros((out.shape), dtype=bool)

            out[b0_pos.reshape(-1)] = False
            bmat_i = bmat[~out.reshape(-1)]
            dwi_i = dwi_i[~out.reshape(-1)]
            n_i = dwi_i.size
            ndof_i = n_i - bmat_i.shape[1]

            # ------------------------------------------------------------
            # Initial WLLS estimate
            # ------------------------------------------------------------
            try:
                dt_i = np.linalg.lstsq(
                    bmat_i,
                    np.log(np.clip(dwi_i, 1, None)),
                    rcond=None,
                )[0]
            except Exception:
                dt_i = np.full((bmat_i.shape[1], 1), minZero)

            # ensure consistent shape
            try:
                dt_i = np.asarray(dt_i, dtype=np.float64).reshape(-1, 1)
            except Exception:
                dt_i = np.full((bmat_i.shape[1], 1), minZero, dtype=np.float64)

            # Compute initial weights safely
            with np.errstate(all="ignore"):
                z = np.matmul(bmat_i, dt_i)

            z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
            z = np.clip(z, -50.0, 50.0)

            w = highprecisionexp(z)
            w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
            w = np.clip(w, 0.0, 1e6)

            try:
                dt_i = np.linalg.lstsq(
                    (bmat_i * np.tile(w, (1, nparam))),
                    (np.log(np.clip(dwi_i, 1, None)).reshape((dwi_i.shape[0], 1)) * w),
                    rcond=None,
                )[0]
            except Exception:
                dt_i = np.full((bmat_i.shape[1], 1), minZero)

            try:
                dt_i = np.asarray(dt_i, dtype=np.float64).reshape(-1, 1)
            except Exception:
                dt_i = np.full((bmat_i.shape[1], 1), minZero, dtype=np.float64)

            # Initial predicted signal
            with np.errstate(all="ignore"):
                z = np.matmul(bmat_i, dt_i)

            z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
            z = np.clip(z, -50.0, 50.0)

            dwi_hat = highprecisionexp(z)
            dwi_hat = np.nan_to_num(dwi_hat, nan=0.0, posinf=1e12, neginf=0.0)
            dwi_hat[dwi_hat < 1] = 1

            # Initial goodness-of-fit
            residu = np.log(np.clip(dwi_i.reshape((dwi_i.shape[0], 1)), 1, None)) - np.log(dwi_hat)
            residu_ = dwi_i.reshape((dwi_i.shape[0], 1)) - dwi_hat

            try:
                chi2 = np.sum((residu_ * residu_) / np.square(sigma)) / (ndof_i) - 1
            except Exception:
                chi2 = minZero

            try:
                gof = np.abs(chi2) < 3 * np.sqrt(2 / ndof_i)
            except Exception:
                gof = True  # If ndof_i == 0, treat as converged

            gof2 = gof

            # ------------------------------------------------------------
            # Iterative reweighting procedure
            # ------------------------------------------------------------
            iter = 0
            dt_imin1 = dt_i.copy()   # <-- critical fix: always initialized
            np.seterr(divide="raise", invalid="raise")

            while (not gof) and (iter < maxiter):
                # Save previous iterate BEFORE computing new one
                dt_imin1 = dt_i.copy()

                try:
                    C = (
                        np.sqrt(n_i / (n_i - nparam))
                        * 1.4826
                        * np.median(np.abs(residu_ - np.median(residu_)))
                        / dwi_hat
                    )
                except Exception:
                    C = np.full(dwi_hat.shape, minZero)

                try:
                    GMM = np.square(C) / np.square(np.square(residu) + np.square(C))
                except Exception:
                    GMM = np.full(C.shape, minZero)

                w = np.sqrt(GMM) * dwi_hat
                w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
                w = np.clip(w, 0.0, 1e6)

                try:
                    dt_i = np.linalg.lstsq(
                        (bmat_i * np.tile(w, (1, nparam))),
                        (np.log(np.clip(dwi_i, 1, None)).reshape((dwi_i.shape[0], 1)) * w),
                        rcond=None,
                    )[0]
                except Exception:
                    dt_i = np.full((bmat_i.shape[1], 1), minZero)

                try:
                    dt_i = np.asarray(dt_i, dtype=np.float64).reshape(-1, 1)
                except Exception:
                    dt_i = np.full((bmat_i.shape[1], 1), minZero, dtype=np.float64)

                # Update predicted signal safely
                with np.errstate(all="ignore"):
                    z = np.matmul(bmat_i, dt_i)

                z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
                z = np.clip(z, -50.0, 50.0)

                dwi_hat = highprecisionexp(z)
                dwi_hat = np.nan_to_num(dwi_hat, nan=0.0, posinf=1e12, neginf=0.0)
                dwi_hat[dwi_hat < 1] = 1

                residu = np.log(np.clip(dwi_i.reshape((dwi_i.shape[0], 1)), 1, None)) - np.log(dwi_hat)
                residu_ = dwi_i.reshape((dwi_i.shape[0], 1)) - dwi_hat

                # Convergence check
                iter = iter + 1
                gof = np.linalg.norm(dt_i - dt_imin1) < np.linalg.norm(dt_i) * convcrit

            np.seterr(**defaultErrorState)

            # ------------------------------------------------------------
            # Outlier detection
            # ------------------------------------------------------------
            if ~gof2:
                try:
                    # Use weighted design matrix directly for leverage
                    # Keep leverage weights conservative; this is only for hat-diagonal estimation
                    w_lev = np.clip(dwi_hat.reshape(-1, 1), 0.0, 1e3)
                    w_lev = np.nan_to_num(w_lev, nan=0.0, posinf=1e3, neginf=0.0)

                    A_w = bmat_i * w_lev
                    A_w = np.nan_to_num(A_w, nan=0.0, posinf=0.0, neginf=0.0)

                    # Leverage from QR: diag(H) = row-wise squared norm of Q
                    with np.errstate(all="ignore"):
                        Q, _ = np.linalg.qr(A_w, mode="reduced")

                    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
                    lev = np.sum(Q * Q, axis=1).reshape(-1, 1)

                    lev = np.nan_to_num(lev, nan=minZero, posinf=1.0, neginf=minZero)
                    lev = np.clip(lev, 0.0, 1.0)

                except Exception:
                    lev = np.full(residu.shape, minZero)

                try:
                    lowerbound_linear = -bounds * np.lib.scimath.sqrt(1 - lev) * sigma / dwi_hat
                    lowerbound_linear = np.nan_to_num(
                        lowerbound_linear, nan=minZero, posinf=minZero, neginf=minZero
                    )
                except Exception:
                    lowerbound_linear = np.full(lev.shape, minZero)

                try:
                    upperbound_nonlinear = bounds * np.lib.scimath.sqrt(1 - lev) * sigma
                    upperbound_nonlinear = np.nan_to_num(
                        upperbound_nonlinear, nan=minZero, posinf=minZero, neginf=minZero
                    )
                except Exception:
                    upperbound_nonlinear = np.full(lev.shape, minZero)

                tmp = np.zeros(residu.shape, dtype=bool, order="F")
                tmp[residu < lowerbound_linear] = True
                tmp[residu > upperbound_nonlinear] = True
                tmp[lev > leverage] = False

                tmp2 = np.ones(b.shape, dtype=bool, order="F")
                tmp2[~out.reshape(-1)] = tmp
                tmp2[b0_pos] = False
                reject = tmp2
            else:
                tmp2 = np.zeros(b.shape, dtype=bool, order="F")
                tmp2[out.reshape(-1)] = True
                reject = tmp2
            # ------------------------------------------------------------
            # Robust parameter estimation
            # ------------------------------------------------------------
            keep = ~reject.reshape(-1)
            bmat_i = bmat[keep, :]
            dwi_i = dwi[keep]

            try:
                dt_ = np.linalg.lstsq(
                    bmat_i,
                    np.log(np.clip(dwi_i, 1, None)),
                    rcond=None,
                )[0]
                dt_ = np.asarray(dt_).reshape(-1, 1)
            except Exception:
                dt_ = np.full((bmat_i.shape[1], 1), minZero)

            with np.errstate(all="ignore"):
                z = np.matmul(bmat_i, dt_)

            z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
            z = np.clip(z, -50.0, 50.0)

            w = highprecisionexp(z)
            w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
            w = np.clip(w, 0.0, 1e6).reshape(-1, 1)

            try:
                # stable weighted least squares via row scaling
                A_w = bmat_i * w
                y_w = np.log(np.clip(dwi_i, 1, None)).reshape(-1, 1) * w

                A_w = np.nan_to_num(A_w, nan=0.0, posinf=0.0, neginf=0.0)
                y_w = np.nan_to_num(y_w, nan=0.0, posinf=0.0, neginf=0.0)

                dt = np.linalg.lstsq(A_w, y_w, rcond=None)[0]
            except Exception:
                dt = np.full((bmat_i.shape[1], 1), minZero)

            return reject.reshape(-1), dt.reshape(-1)

            #     with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            #         dwi_hat = highprecisionexp(np.matmul(bmat_i, dt_i))

            #     dwi_hat = np.nan_to_num(dwi_hat, nan=0.0, posinf=0.0, neginf=0.0)

            #     dwi_hat[dwi_hat < 1] = 1
            #     residu = np.log(dwi_i.reshape((dwi_i.shape[0], 1))) - np.log(dwi_hat)
            #     residu_ = dwi_i.reshape((dwi_i.shape[0], 1)) - dwi_hat
            #     # Convergence check
            #     iter = iter + 1
            #     gof = np.linalg.norm(dt_i - dt_imin1) < np.linalg.norm(dt_i) * convcrit
            # np.seterr(**defaultErrorState)
            # # Outlier detection
            # if ~gof2:
            #     try:
            #         lev = np.diag(
            #             np.matmul(
            #                 bmat_i,
            #                 np.linalg.lstsq(
            #                     np.matmul(
            #                         np.transpose(bmat_i),
            #                         np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i),
            #                     ),
            #                     np.matmul(
            #                         np.transpose(bmat_i),
            #                         np.diag(np.square(w.reshape(-1))),
            #                     ),
            #                     rcond=None,
            #                 )[0],
            #             )
            #         )
            #     except:  # noqa: E722
            #         lev = np.full(residu.shape, minZero)
            #     lev = lev.reshape((lev.shape[0], 1))
            #     try:
            #         lowerbound_linear = -bounds * np.lib.scimath.sqrt(1 - lev) * sigma / dwi_hat
            #     except:  # noqa: E722
            #         lowerbound_linear = np.full(lev.shape, minZero)
            #     try:
            #         upperbound_nonlinear = bounds * np.lib.scimath.sqrt(1 - lev) * sigma
            #     except:  # noqa: E722
            #         upperbound_nonlinear = np.full(lev.shape, minZero)
            #     tmp = np.zeros(residu.shape, dtype=bool, order="F")
            #     tmp[residu < lowerbound_linear] = True
            #     tmp[residu > upperbound_nonlinear] = True
            #     tmp[lev > leverage] = False
            #     tmp2 = np.ones(b.shape, dtype=bool, order="F")
            #     tmp2[~out.reshape(-1)] = tmp
            #     tmp2[b0_pos] = False
            #     reject = tmp2
            # else:
            #     tmp2 = np.zeros(b.shape, dtype=bool, order="F")
            #     tmp2[out.reshape(-1)] = True
            #     reject = tmp2
            # # Robust parameter estimation
            # keep = ~reject.reshape(-1)
            # bmat_i = bmat[keep, :]
            # dwi_i = dwi[keep]
            # try:
            #     dt_ = np.linalg.lstsq(bmat_i, np.log(dwi_i), rcond=None)[0]
            # except:  # noqa: E722
            #     dt_ = np.full((bmat_i.shape[1], 1), minZero)
            # with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            #     w = highprecisionexp(np.matmul(bmat_i, dt_))
            # try:
            #     dt = np.linalg.lstsq(
            #         (bmat_i * np.tile(w.reshape((len(w), 1)), (1, nparam))),
            #         (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w.reshape((len(w), 1))),
            #         rcond=None,
            #     )[0]
            # except:  # noqa: E722
            #     dt = np.full((bmat_i.shape[1], 1), minZero)
            # return reject.reshape(-1), dt.reshape(-1)  # , fa, md

        inputs = tqdm(
            range(nvox),
            desc="IRLLS Outlier Detection",
            bar_format="{desc}: [{percentage:0.0f}%]",
            unit="vox",
            ncols=tqdmWidth,
        )
        (reject, dt) = zip(
            *Parallel(n_jobs=self.workers, prefer="processes")(
                delayed(outlierHelper)(dwi[:, i], bmat, sigma[i, 0], b, b0_pos) for i in inputs
            )
        )
        # for i in inputs:
        #     reject[:,i], dt[:,i] = outlierHelper(dwi[:, i], bmat, sigma[i,0], b, b0_pos)
        dt = np.array(dt)
        # self.dt = dt
        # Unscaling
        if scaling:
            dt[1, :] = dt[1, :] + np.log(sc / 1000)
        # Unvectorizing
        reject = vectorize(np.array(reject).T, self.mask)
        return reject, dt.T

    def tensorReorder(self, dwiType: str) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """Reorders tensors in DT to those of MRTRIX in accordance to
        the table below.

        Parameters
        ----------
        dwiType: str, {'dti', 'dki'}
            Indicates whether image is DTI or DKI.

        Returns
        -------
        DT: ndarray(dtype=float)
            4D image containing DT tensor.
        KT: ndarray(dtype=float)
            4D image containing KT tensor.

        Examples
        --------
        dt = dwi.tensorReorder()

        Notes
        -----
        MRTRIX3 and Designer tensors are described below.

        .. code-block:: none

            MRTRIX3 Tensors                     DESIGNER Tensors
            ~~~~~~~~~~~~~~~                     ~~~~~~~~~~~~~~~~

            0   D0      1   1                       1   1
            1   D1      2   2                       1   2
            2   D2      3   3                       1   3
            3   D3      1   2                       2   2
            4   D4      1   3                       2   3
            5   D5      2   3                       3   3

            6   K0      1   1   1   1               1   1   1   1
            7   K1      2   2   2   2               1   1   1   2
            8   K2      3   3   3   3               1   1   1   3
            9   K3      1   1   1   2               1   1   2   2
            10  K4      1   1   1   3               1   1   2   3
            11  K5      1   2   2   2               1   1   3   3
            12  K6      1   3   3   3               1   2   2   2
            13  K7      2   2   2   3               1   2   2   3
            14  K8      2   3   3   3               1   2   3   3
            15  K9      1   1   2   2               1   3   3   3
            16  K10     1   1   3   3               2   2   2   2
            17  K11     2   2   3   3               2   2   2   3
            18  K12     1   1   2   3               2   2   3   3
            19  K13     1   2   2   3               2   3   3   3
            20  K14     1   2   3   3               3   3   3   3

            Value Assignment
            ~~~~~~~~~~~~~~~~

            MRTRIX3         DESIGNER
            ~~~~~~~         ~~~~~~~~
                0               0
                1               3
                2               5
                3               1
                4               2
                5               4

                6               6
                7               16
                8               20
                9               7
                10              8
                11              12
                12              15
                13              17
                14              19
                15              9
                16              11
                17              18
                18              10
                19              13
                20              14
        """
        if self.dt is None:
            raise Exception("Please run dwi.fit() to generate a tensor prior to reordering tensors.")

        if dwiType == "dti":
            dt = np.zeros((6, self.dt.shape[1]))
            dt[0, :] = self.dt[0, :]  # D0
            dt[1, :] = self.dt[3, :]  # D1
            dt[2, :] = self.dt[5, :]  # D2
            dt[3, :] = self.dt[1, :]  # D3
            dt[4, :] = self.dt[2, :]  # D4
            dt[5, :] = self.dt[4, :]  # D5
            DT = vectorize(dt[0:6, :], self.mask)
            KT = None
        if dwiType == "dki":
            dt = np.zeros(self.dt.shape)
            dt[0, :] = self.dt[0, :]  # D0
            dt[1, :] = self.dt[3, :]  # D1
            dt[2, :] = self.dt[5, :]  # D2
            dt[3, :] = self.dt[1, :]  # D3
            dt[4, :] = self.dt[2, :]  # D4
            dt[5, :] = self.dt[4, :]  # D5
            dt[6, :] = self.dt[6, :]  # K0
            dt[7, :] = self.dt[16, :]  # K1
            dt[8, :] = self.dt[20, :]  # K2
            dt[9, :] = self.dt[7, :]  # K3
            dt[10, :] = self.dt[8, :]  # K4
            dt[11, :] = self.dt[12, :]  # K5
            dt[12, :] = self.dt[15, :]  # K6
            dt[13, :] = self.dt[17, :]  # K7
            dt[14, :] = self.dt[19, :]  # K8
            dt[15, :] = self.dt[9, :]  # K9
            dt[16, :] = self.dt[11, :]  # K10
            dt[17, :] = self.dt[18, :]  # K11
            dt[18, :] = self.dt[10, :]  # K12
            dt[19, :] = self.dt[13, :]  # K13
            dt[20, :] = self.dt[14, :]  # K14
            DT = vectorize(dt[0:6, :], self.mask)
            KT = vectorize(dt[6:21, :], self.mask)
        return DT, KT

    def irllsviolmask(self, reject: np.ndarray[bool]) -> np.ndarray[float]:
        """Computes 3D violation mask of outliers detected from IRLLS
        method.

        Parameters
        ----------
        reject: ndarray(dtype=bool)
            4D input outlier map from IRLLS.

        Returns
        -------
        propviol: ndarray(dtype=float)
            3D mask where voxel value is the percentage of directional
            violations.

        Examples
        --------
        mask = dwi.irllsviolmask(outliers)
        """
        img = vectorize(reject, self.mask)
        (ndwi, nvox) = img.shape
        b = np.array(self.grad[:, 3])
        b = np.reshape(b, (len(b), 1))
        b_pos = ~(b < 0.01).reshape(-1)
        img = img[b_pos, :]
        propViol = np.sum(img, axis=0).astype(int) / np.sum(b_pos)
        propViol = vectorize(propViol, self.mask)
        return propViol


def fit_regime(
    input: str,
    output: str,
    prefix: str = None,
    suffix: str = None,
    ext: str = None,
    irlls: bool = True,
    akc: bool = True,
    qcpath: Union[str, None] = None,
    fit_constraints: List[int] = [0, 1, 0],
    l_max: Union[int, None] = None,
    rectify: bool = True,
    res: str = "med",
    n_fibers: int = 5,
    mask: Union[str, None] = None,
    window: bool = False,
    nthreads: Union[int, None] = None,
) -> None:
    """Performs the entire tensor fitting regime and writes out maps.
    Uses auto-detections methods to determine the types of protocols
    encoded by DWI and extract their metrics.

    Parameters
    ----------
    input : str
        Path to DWI
    output : str
        Output directory to write all outputs
    prefix : str
        Prefix to append to output file names.
        (Default: None)
    suffix : str
        Suffix to append to output file names.
        (Default: None)
    ext : str
        Specify output image extension type.
        (Default: None)
    irlls : bool
        Specify whether to perform IRLLS outlier detection.
        (Default: True)
    akc : bool
        Specify whether to perform brute-forced AKC correction.
        (Default: True)
    qcpath : str
        Specify output directory to write QC metrics.
        (Default: None)
    fit_constraints : list
        List of 3 bool elements specifying which fit contraints to
        use. See DWI.createConstraints() on usage.
        (Default: [0, 1, 0])
    l_max : int
        Maximum spherical harminic degree for FBI/FBWM fit.
        (Default: None)
    rectify : bool
        Specify whether to rectify FBI fODF
        (Default: True)
    mask : str
        Path to brain mask
        (Default: None)
    window : bool
        Window metric map within biological limits
        (Default: False)
    nthreads : int
        Number of workers to use in processing. Default value uses all
        available workers.
        (Default: None)
    """
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""
    if ext is None:
        ext = ".nii"
    img = DWI(input, mask=mask, nthreads=nthreads)
    protocols = img.tensorType()
    print("Protocol(s) detected: {}".format(", ".join([x.upper() for x in protocols])))
    fname_dti = {}
    fname_dki = {}
    fname_wmti = {}
    fname_fbi = {}
    fname_tensor = {}
    fname_outliers = {}
    fname_tractography = {}
    for key, value in dwi_fnames._dti_.items():
        fname_dti[key] = prefix + value + suffix + ext
    for key, value in dwi_fnames._dki_.items():
        fname_dki[key] = prefix + value + suffix + ext
    for key, value in dwi_fnames._wmti_.items():
        fname_wmti[key] = prefix + value + suffix + ext
    for key, value in dwi_fnames._fbi_.items():
        fname_fbi[key] = prefix + value + suffix + ext
    for key, value in dwi_fnames._tensor_.items():
        fname_tensor[key] = prefix + value + suffix + ext
    for key, value in dwi_fnames._outliers_.items():
        fname_outliers[key] = prefix + value + suffix + ext
    for key, value in dwi_fnames._tractography_.items():
        fname_tractography[key] = prefix + value + suffix + ".fib"
    if irlls:
        if img.isdki():
            outliers, dt_est = img.irlls(mode="DKI", excludeb0=True)
        else:
            outliers, dt_est = img.irlls(mode="DTI", excludeb0=True)

        if qcpath:
            if op.exists(qcpath):
                outlier_full = op.join(qcpath, fname_outliers["IRLLS"])
                print(outlier_full)
                outlier_plot_full = op.join(qcpath, prefix + "irlls_outliers_plot" + suffix + ".png")
                bvals_outlier_full = op.join(qcpath, prefix + "irlls_outliers_shells" + suffix + ".bval")
                if img.isdki():
                    bvals_outlier = img.getBvals()[img.idxdki()].astype(int)
                else:
                    bvals_outlier = img.getBvals()[img.idxdti()].astype(int)
                bvals_outlier = bvals_outlier * 1000
                writeNii(outliers, img.hdr, outlier_full)
                np.savetxt(bvals_outlier_full, bvals_outlier, newline=" ", fmt="%d")
                if mask:
                    outlierplot.plot(
                        input=outlier_full,
                        output=outlier_plot_full,
                        bval=bvals_outlier_full,
                        mask=mask,
                    )
                else:
                    outlierplot.plot(
                        input=outlier_full,
                        output=outlier_plot_full,
                        bval=bvals_outlier_full,
                        mask=None,
                    )
                os.remove(bvals_outlier_full)
    if irlls:
        img.fit(fit_constraints, reject=outliers)
    else:
        img.fit(fit_constraints)

    if akc and img.isdki():
        akc_out = img.akcoutliers()
        img.akccorrect(akc_out)
        if qcpath:
            writeNii(akc_out, img.hdr, op.join(qcpath, fname_outliers["AKC"]))
    if "dki" in img.tensorType():
        tensorType = "dki"
    else:
        tensorType = "dti"
    DT, KT = img.tensorReorder(tensorType)
    if tensorType == "dki":
        writeNii(DT, img.hdr, op.join(output, fname_tensor["DT"]))
        writeNii(KT, img.hdr, op.join(output, fname_tensor["KT"]))
    else:
        writeNii(DT, img.hdr, op.join(output, fname_tensor["DT"]))
    # DTI Parameters
    if img.isdti():
        md, rd, ad, fa, fe, trace = img.extractDTI()
        writeNii(md, img.hdr, op.join(output, fname_dti["md"]), range=th.__md__, clip=window)
        writeNii(rd, img.hdr, op.join(output, fname_dti["rd"]), range=th.__rd__, clip=window)
        writeNii(ad, img.hdr, op.join(output, fname_dti["ad"]), range=th.__ad__, clip=window)
        writeNii(fa, img.hdr, op.join(output, fname_dti["fa"]), range=th.__fa__, clip=window)
        writeNii(fe, img.hdr, op.join(output, fname_dti["fe"]))
        writeNii(trace, img.hdr, op.join(output, fname_dti["trace"]))
        dtimodel = odf.odfmodel(dt=op.join(output, fname_tensor["DT"]), mask=mask, l_max=2, res=res)
        dti_odfs = dtimodel.dtiodf()
        dti_sh = dtimodel.odf2sh(dti_odfs)
        dtimodel.savenii(dti_sh, op.join(output, fname_dti["odf"]))
        dsistudio.makefib(
            input=op.join(output, fname_dti["odf"]),
            output=op.join(output, fname_tractography["dti"]),
            map=op.join(output, fname_dti["fa"]),
            mask=mask,
            n_fibers=n_fibers,
            scale=0.5,
            other_maps=[
                op.join(output, fname_dti["md"]),
                op.join(output, fname_dti["rd"]),
                op.join(output, fname_dti["ad"]),
            ],
        )
    if img.isdki():
        # DKI Parameters
        mk, rk, ak, kfa, mkt, trace = img.extractDKI()
        writeNii(mk, img.hdr, op.join(output, fname_dki["mk"]), range=th.__mk__, clip=window)
        writeNii(rk, img.hdr, op.join(output, fname_dki["rk"]), range=th.__rk__, clip=window)
        writeNii(ak, img.hdr, op.join(output, fname_dki["ak"]), range=th.__ak__, clip=window)
        writeNii(kfa, img.hdr, op.join(output, fname_dki["kfa"]), range=th.__kfa__, clip=window)
        writeNii(mkt, img.hdr, op.join(output, fname_dki["mkt"]))
        writeNii(trace, img.hdr, op.join(output, fname_dki["trace"]))
        # WMTI Parameters
        awf, eas_ad, eas_rd, eas_tort, ias_da = img.extractWMTI()
        writeNii(awf, img.hdr, op.join(output, fname_wmti["awf"]), range=th.__wawf__, clip=window)
        writeNii(eas_ad, img.hdr, op.join(output, fname_wmti["eas_ad"]), range=th.__eas_ad__, clip=window)
        writeNii(eas_rd, img.hdr, op.join(output, fname_wmti["eas_rd"]), range=th.__eas_rd__, clip=window)
        writeNii(eas_tort, img.hdr, op.join(output, fname_wmti["eas_tort"]), range=th.__tort__, clip=window)
        writeNii(ias_da, img.hdr, op.join(output, fname_wmti["ias_da"]), range=th.__ias_da__, clip=window)
        dkimodel = odf.odfmodel(
            dt=op.join(output, fname_tensor["DT"]),
            kt=op.join(output, fname_tensor["KT"]),
            mask=mask,
            l_max=6,
            res=res,
        )
        dki_odfs = dkimodel.dkiodf(fa_t=0.90)
        dki_sh = dkimodel.odf2sh(dki_odfs)
        dkimodel.savenii(dki_sh, op.join(output, fname_dki["odf"]))
        dsistudio.makefib(
            input=op.join(output, fname_dki["odf"]),
            output=op.join(output, fname_tractography["dki"]),
            map=op.join(output, fname_dti["fa"]),
            mask=mask,
            n_fibers=n_fibers,
            scale=0.5,
            other_maps=[
                op.join(output, fname_dti["md"]),
                op.join(output, fname_dti["rd"]),
                op.join(output, fname_dti["ad"]),
                op.join(output, fname_dki["mk"]),
                op.join(output, fname_dki["rk"]),
                op.join(output, fname_dki["ak"]),
                op.join(output, fname_dki["kfa"]),
                op.join(output, fname_wmti["awf"]),
                op.join(output, fname_wmti["eas_ad"]),
                op.join(output, fname_wmti["eas_rd"]),
                op.join(output, fname_wmti["eas_tort"]),
                op.join(output, fname_wmti["ias_da"]),
            ],
        )
    if img.isfbi():
        if img.isfbwm():
            (
                zeta,
                faa,
                sph,
                sph_mrtrix,
                min_awf,
                Da,
                De_mean,
                De_ax,
                De_rad,
                De_fa,
                min_cost,
                min_cost_fn,
            ) = img.fbi(l_max=l_max, fbwm=True, rectify=rectify)
            writeNii(zeta, img.hdr, op.join(output, fname_fbi["zeta"]), range=th.__zeta__, clip=window)
            writeNii(faa, img.hdr, op.join(output, fname_fbi["faa"]), range=th.__faa__, clip=window)
            writeNii(np.real(sph), img.hdr, op.join(output, fname_fbi["odf"]))
            writeNii(np.real(sph_mrtrix), img.hdr, op.join(output, fname_fbi["odf_mrtrix"]))
            writeNii(min_awf, img.hdr, op.join(output, fname_fbi["awf"]), range=th.__fawf__, clip=window)
            writeNii(Da, img.hdr, op.join(output, fname_fbi["Da"]), range=th.__da__, clip=window)
            writeNii(De_mean, img.hdr, op.join(output, fname_fbi["De_mean"]), range=th.__de_mean__, clip=window)
            writeNii(De_ax, img.hdr, op.join(output, fname_fbi["De_ax"]), range=th.__de__ax__, clip=window)
            writeNii(De_rad, img.hdr, op.join(output, fname_fbi["De_rad"]), range=th.__de_rad__, clip=window)
            writeNii(De_fa, img.hdr, op.join(output, fname_fbi["fae"]), range=th.__de_fa__, clip=window)
            writeNii(min_cost, img.hdr, op.join(output, fname_fbi["min_cost"]))
            writeNii(min_cost_fn, img.hdr, op.join(output, fname_fbi["min_cost_fn"]))
            dsistudio.makefib(
                input=op.join(output, fname_fbi["odf_mrtrix"]),
                output=op.join(output, fname_tractography["fbi"]),
                map=op.join(output, fname_fbi["faa"]),
                mask=mask,
                n_fibers=n_fibers,
                scale=1,
                other_maps=[
                    op.join(output, fname_dti["md"]),
                    op.join(output, fname_dti["rd"]),
                    op.join(output, fname_dti["ad"]),
                    op.join(output, fname_dki["mk"]),
                    op.join(output, fname_dki["rk"]),
                    op.join(output, fname_dki["ak"]),
                    op.join(output, fname_dki["kfa"]),
                    op.join(output, fname_wmti["awf"]),
                    op.join(output, fname_wmti["eas_ad"]),
                    op.join(output, fname_wmti["eas_rd"]),
                    op.join(output, fname_wmti["eas_tort"]),
                    op.join(output, fname_wmti["ias_da"]),
                    op.join(output, fname_fbi["zeta"]),
                    op.join(output, fname_fbi["awf"]),
                    op.join(output, fname_fbi["Da"]),
                    op.join(output, fname_fbi["De_mean"]),
                    op.join(output, fname_fbi["De_ax"]),
                    op.join(output, fname_fbi["De_rad"]),
                    op.join(output, fname_fbi["fae"]),
                ],
            )
        else:
            (
                zeta,
                faa,
                sph,
                sph_mrtrix,
                min_awf,
                Da,
                De_mean,
                De_ax,
                De_rad,
                De_fa,
                min_cost,
                min_cost_fn,
            ) = img.fbi(l_max=l_max, fbwm=False, rectify=rectify)
            writeNii(zeta, img.hdr, op.join(output, fname_fbi["zeta"]))
            writeNii(faa, img.hdr, op.join(output, fname_fbi["faa"]))
            writeNii(np.real(sph), img.hdr, op.join(output, fname_fbi["odf"]))
            writeNii(np.real(sph_mrtrix), img.hdr, op.join(output, fname_fbi["odf_mrtrix"]))
            dsistudio.makefib(
                input=op.join(output, fname_fbi["odf"]),
                output=op.join(output, fname_tractography["fbi"]),
                map=op.join(output, fname_fbi["faa"]),
                mask=mask,
                n_fibers=n_fibers,
                scale=1,
                other_maps=[op.join(output, fname_fbi["zeta"])],
            )
