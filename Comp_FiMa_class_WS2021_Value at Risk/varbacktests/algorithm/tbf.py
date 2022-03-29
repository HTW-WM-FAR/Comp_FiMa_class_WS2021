from scipy.stats import binom, norm, chi2
import numpy as np
import math
from varbacktests.algorithm.tbfi import Tbfi
from varbacktests.algorithm.pof import Pof


class Tbf:
    def __init__(self) -> None:
        self.tbfi = Tbfi()
        self.pof = Pof()

    def test(self, failures, pVaR, TestLevel=0.95):
        output = {}

        try:
            n = failures.index(1)
        except:
            n = 0
        N = len(failures)

        pof_result = self.pof.test(failures, pVaR, TestLevel)
        tbfi_result = self.tbfi.test(failures, pVaR, TestLevel)

        LRatioPOF = pof_result["LR"]
        LRatioTBFI = tbfi_result["LR"]

        dof = tbfi_result["x"] + 1
        LR = LRatioPOF + LRatioTBFI
        LRThres = chi2.ppf(TestLevel, dof)
        tInternal = LR < LRThres
        PValue = chi2.cdf(LR, dof)

        output["LR"] = LR
        output["pVal"] = PValue
        output["N"] = len(failures)
        output["x"] = tbfi_result["x"]
        output["result"] = "accept" if tInternal else "reject"

        return output
