from scipy.stats import binom, norm, chi2
import numpy as np
import math


class CCI:
    def __init__(self):
        pass

    def notList(self, a):
        return [abs(i - 1) for i in a]

    def andList(self, a, b):
        return [max(i, j) for i, j in zip(a, b)]

    def test(self, failures, pVaR, TestLevel=0.95):
        output = {}
        N = len(failures)
        x = sum(failures)

        n00 = sum(
            self.andList(self.notList(failures[0 : N - 1]), self.notList(failures[1:N]))
        )

        n10 = sum(self.andList(failures[0 : N - 1], self.notList(failures[1:N])))

        n01 = sum(self.andList(self.notList(failures[0 : N - 1]), failures[1:N]))

        n11 = sum(self.andList(failures[0 : N - 1], failures[1:N]))

        LogLNum = 0

        if (n00 + n10 > 0) and (n01 + n11 > 0):
            pUC = (n01 + n11) / (n00 + n10 + n01 + n11)
            LogLNum = (n00 + n10) * math.log1p(1 - pUC) + (n01 + n11) * math.log1p(pUC)

        LogLDen = 0
        # log(LDen) = log((1-p01)^n00 p01^n01 (1-p11)^n10 p11^n11)
        if n00 > 0 and n01 > 0:
            p01 = n01 / (n00 + n01)
            LogLDen = LogLDen + n00 * math.log1p(1 - p01) + n01 * math.log1p(p01)
        if n10 > 0 and n11 > 0:
            p11 = n11 / (n10 + n11)
            LogLDen = LogLDen + n10 * math.log1p(1 - p11) + n11 * math.log1p(p11)

        LR = 2 * LogLNum - 2 * LogLDen

        dof = 1
        LRThres = chi2.ppf(TestLevel, dof)
        tInternal = LR < LRThres

        output["LR"] = LR
        output["pVal"] = chi2.cdf(LR, dof)
        output["N"] = N
        output["x"] = x
        output["result"] = "accept" if tInternal else "reject"
        output["n00"] = n00
        output["n10"] = n10
        output["n01"] = n01
        output["n11"] = n11

        return output
