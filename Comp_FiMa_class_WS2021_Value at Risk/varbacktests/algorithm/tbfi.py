from scipy.stats import binom, norm, chi2
import numpy as np
import math


class Tbfi:
    def __init__(self):
        pass

    def find_non_zero(self, a):
        res = []
        for i, item in enumerate(a):
            if item > 0:
                res.append(i)
        return res

    def diff(self, a):
        res = [
            0,
        ]
        for i in range(1, len(a)):
            res.append(a[i] - a[i - 1])
        return res

    def test(self, failures, pVaR, TestLevel=0.95):
        output = {}

        N = len(failures)
        x = sum(failures)

        tbf = self.diff(self.find_non_zero(failures))

        if x > 0:
            LR = 0
            for i in range(0, x):
                n = tbf[i]
                if n > 1:
                    LR = LR - 2 * (
                        math.log(pVaR)
                        + (n - 1) * math.log(1 - pVaR)
                        + n * math.log(n)
                        - (n - 1) * math.log(n - 1)
                    )
                else:
                    LR = LR - 2 * math.log(pVaR)

            dof = x
            LRThres = chi2.ppf(TestLevel, dof)
            tInternal = LR < LRThres

            output["LR"] = LR
            output["pVal"] = chi2.cdf(LR, dof)
            output["N"] = N
            output["x"] = x
            output["result"] = "accept" if tInternal else "reject"
            output["dist"] = np.quantile(tbf, [0, 0.25, 0.5, 0.75, 1])

            return output
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

            output["LR"] = 0
            output["pVal"] = 0
            output["N"] = 0
            output["x"] = 0

            if N + 1 > 1 / pVaR:
                tInternal = LRBound < LRThres
                output["result"] = "accept" if tInternal else "reject"
                output["LR"] = LRBound
                output["pVal"] = chi2.cdf(LRBound, dof)
                output["dist"] = np.quantile(tbf, [0, 0.25, 0.5, 0.75, 1])

            else:
                output["result"] = "accept"
                output["dist"] = np.quantile(tbf, [0, 0.25, 0.5, 0.75, 1])

            return output
