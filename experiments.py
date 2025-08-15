import numpy as np

def positivify_coeffmats(coeffmats: list[np.ndarray], tolerance: float = 1e-6):

    r"""For each MO coefficient matrix in coeffmatlist,
            (which consists of a numpy 2D array with MOs as the columns)
            modify the signs of the MOs so the greatest coeff is positive.
            In order for this to be stable, given there are often equal magnitude coeffs,
            any coeffs within `tolerance` of the largest magnitude coeff are treated as
            equivalent and the one with the lowest AO index is chosen to be the positive one.
        :param coeffmats: `list[np.ndarray]` of coefficient matrices
        :param tolerance: `float` difference regarded as degenerate with the largest value.
        Modifies coeffmats in-place.
    """

    for coeffmat in coeffmats:
        for i,coeffvec in enumerate(coeffmat.T):
            sortcoeffs = sorted([(abs(x),ind,x) for ind,x in enumerate(coeffvec)])
            maxabs = sortcoeffs[-1][0]
            #Filter all those within 1e-6 of max amplitude
            filtcoeffs = [tup for tup in sortcoeffs if abs(tup[0]-maxabs)<1e-6]
            #Now pick the one with the lowest index
            sortfiltcoeffs = sorted(filtcoeffs,key= lambda tup: tup[1])
            _,ind,coeff = sortfiltcoeffs[0]
            if coeff<0:
                coeffmat[:,ind]*=-1


A = np.array([
    [-0.20,  0.10, -0.50],
    [ 0.90, -0.70,  0.50],
    [-0.10,  0.60, -0.10],
], dtype=float)  # rows = AOs, cols = MOs

B = np.array([
    [ 0.30,  0.40],
    [-0.60,  0.10],
], dtype=float)

mats = [A, B]

print("Before:\nA=\n", A, "\nB=\n", B)
positivify_coeffmats(mats, tolerance=1e-6)  # modifies A and B in place
print("\nAfter:\nA=\n", A, "\nB=\n", B)