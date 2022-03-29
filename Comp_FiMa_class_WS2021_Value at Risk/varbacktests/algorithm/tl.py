import numpy as np
from scipy.stats import binom, norm


class TL:
    def __init__(self):
        pass

    def test(self, failures, pVaR):
        N = len(failures)
        x = sum(failures)
        ZONE = ["green", "yellow", "red"]

        p = binom.cdf(x, N, pVaR)

        YellowCutOff = 0.95
        RedCutOff = 0.9999

        x = np.array([x])

        zone = np.digitize(x, [0, YellowCutOff, RedCutOff, 999999999])

        output = {}

        output["prob"] = p

        if x[0] == 0:
            output["TypeI"] = 1
        else:
            output["TypeI"] = 1 - p + binom.ppf(x, N, pVaR)

        if zone == 1:
            output["increase"] = 0
        elif zone == 3:
            output["increase"] = 1
        else:
            output["increase"] = self.increaseScalingFactor(N, x[0], pVaR)

        output["N"] = N
        output["x"] = x

        # print(zone[0])

        output["result"] = ZONE[zone[0] - 1]

        # print(output)
        return output

    def increaseScalingFactor(self, N, x, VaRLevel):
        zVaR = norm.ppf(VaRLevel)
        lData = x / N
        zData = norm.ppf(1 - lData)
        Baseline = 3
        delta = max(0, min(1, Baseline * (zVaR / zData - 1)))
        return delta
