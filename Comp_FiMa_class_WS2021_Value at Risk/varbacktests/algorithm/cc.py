from scipy.stats import binom, norm, chi2
import numpy as np
import math
from varbacktests.algorithm.cci import CCI
from varbacktests.algorithm.pof import Pof


class CC:
    def __init__(self):
        self.cci = CCI()
        self.pof = Pof()

    def test(self, failures, pVaR, TestLevel=0.95):
        output = {}
        cci_result = self.cci.test(failures, pVaR, TestLevel)
        pof_result = self.pof.test(failures, pVaR, TestLevel)

        LRatioPOF = pof_result["LR"]
        LRatioCCI = cci_result["LR"]

        LR = LRatioPOF + LRatioCCI

        dof = 2

        LRThres = chi2.ppf(TestLevel, dof)
        tInternal = LR < LRThres
        PValue = chi2.cdf(LR, dof)

        output["LR"] = LR
        output["pVal"] = PValue
        output["N"] = len(failures)
        output["x"] = sum(failures)
        output["result"] = "accept" if tInternal else "reject"

        return output
