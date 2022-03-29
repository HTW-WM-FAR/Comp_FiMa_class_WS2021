from scipy.stats import binom, norm, chi2
import numpy as np
import math


class Pof:
    def __init__(self):
        pass

    def test(self, failures, pVaR, TestLevel=0.95):
        output = {}
        N = len(failures)
        x = sum(failures)

        if x == 0:
            LR = -2 * N * math.log(1 - pVaR)
        elif x < N:
            LR = -2 * (
                (N - x) * math.log(N * (1 - pVaR) / (N - x))
                + x * math.log(N * pVaR / x)
            )
        else:
            LR = -2 * N * math.log(pVaR)

        dof = 1
        LRThres = chi2.ppf(TestLevel, dof)
        tInternal = LR < LRThres

        output["LR"] = LR
        output["pVal"] = chi2.cdf(LR, dof)
        output["N"] = N
        output["x"] = x
        output["result"] = "accept" if tInternal else "reject"

        return output
