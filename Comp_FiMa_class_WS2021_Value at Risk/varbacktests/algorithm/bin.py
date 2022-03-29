from scipy.stats import binom, norm
import numpy as np


class Bin:
    def __init__(self):
        pass

    def test(self, failures, pVaR):
        output = {}
        N = len(failures)
        x = sum(failures)

        Excepted = N * pVaR
        Stdev = np.sqrt(N * pVaR * (1 - pVaR))

        zCrit = norm.ppf(N * pVaR * (1 - pVaR))

        LimLo = Excepted - zCrit * Stdev
        LimHi = Excepted + zCrit * Stdev

        tInternal = LimLo <= x <= LimHi

        zScore = (x - Excepted) / Stdev
        output["zScore"] = zScore
        output["pVal"] = 2 * (1 - norm.cdf(abs(zScore)))
        output["N"] = N
        output["x"] = x
        output["result"] = "accept" if tInternal else "reject"

        return output
