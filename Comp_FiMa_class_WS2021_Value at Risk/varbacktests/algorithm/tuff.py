from scipy.stats import binom, norm, chi2
import numpy as np
import math


class Tuff:
    def __init__(self):
        pass

    def test(self, failures, pVaR, TestLevel=0.95):
        output = {
            "result": "accept",
        }
        try:
            n = failures.index(1)
        except:
            n = 0
            return output

        N = len(failures)

        dof = 1
        LRThres = chi2.ppf(TestLevel, dof)

        if n > 0:
            if n > 1:
                LR = -2 * (
                    (N - n) * math.log(N * (1 - pVaR) / (N - n))
                    + n * math.log(N * pVaR / n)
                )
            else:
                LR = -2 * N * math.log(pVaR)

            tInternal = LR < LRThres

            output["LR"] = LR
            output["pVal"] = chi2.cdf(LR, dof)
            output["N"] = N
            output["n"] = n
            output["result"] = "accept" if tInternal else "reject"
        else:
            dof = 1
            LRThres = chi2.ppf(TestLevel, dof)
            LRBound = (
                -2
                * -2
                * (
                    math.log(pVaR)
                    + (N) * math.log(1 - pVaR)
                    + (N + 1) * math.log(N + 1)
                    - N * math.log(N)
                )
            )

            output["LR"] = None
            output["pVal"] = None
            output["N"] = None
            output["n"] = None

            if N + 1 > 1 / pVaR:
                tInternal = LRBound < LRThres
                output["result"] = "accept" if tInternal else "reject"
                output["LR"] = LRBound
                output["pVal"] = chi2.cdf(LRBound, dof)
            else:
                output["Result"] = True

        return output
