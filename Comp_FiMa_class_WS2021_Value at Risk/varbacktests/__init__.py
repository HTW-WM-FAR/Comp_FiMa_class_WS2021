import numpy as np
from varbacktests.algorithm import Bin, CCI, CC, Pof, TL, Tuff, Tbfi, Tbf
from prettytable import PrettyTable


class VaRBackTest(object):
    def __init__(
        self,
        portfolioData,
        varData,
        portfolioID=["S&P Index"],
        varID=["95"],
        varLevel=[0.95],
    ) -> None:
        self.portfolioData = portfolioData
        self.varData = varData
        self.portfolioID = portfolioID
        self.varID = varID
        self.varLevel = varLevel
        self.failures = self.get_fail_flag()
        self.pVaR = self.get_pVaR(self.varLevel)

        self.backtest_algo = [TL(), Bin(), Pof(), Tuff(), CC(), CCI(), Tbfi(), Tbf()]
        self.table_printer = PrettyTable()

    def get_pVaR(self, VarLevel):
        return [1 - i for i in VarLevel]

    def get_less_index(self, listA, listB):
        res = []
        for i in range(len(listA)):
            if listA[i] < listB[i]:
                res.append(1)
            else:
                res.append(0)
        return res

    def get_fail_flag(self):
        failures = []
        for VaR in self.varData:
            VaR = [-x for x in VaR]
            failures.append(self.get_less_index(self.portfolioData, VaR))
        return failures

    def runtests(self):
        for i in range(len(self.varData)):
            results = {}
            for algo in self.backtest_algo:
                algo_name = str(algo).split(" ")[0].split(".")[-1]
                results[algo_name] = algo.test(self.failures[i], self.pVaR[i])["result"]

            cols = ["PortfolioID", "VarID"]
            cols.extend(list(results.keys()))
            self.table_printer.field_names = cols

            row = [self.portfolioID[0], self.varID[i]]
            row.extend(results.values())

            self.table_printer.add_row(row)
        print(self.table_printer)
        # return results

    def summary(self):
        numVar, numRow = len(self.varData), len(self.varData[0])

        Observations = np.zeros(numVar)
        Failures = np.zeros(numVar)
        FirstFailure = np.zeros(numVar)
        Missing = np.zeros(numVar)
        pVar = np.zeros(numVar)

        for i in range(numVar):
            Observations[i] = numRow
            Failures[i] = sum(self.failures[i])
            try:
                FirstFailure[i] = self.failures[i].index(1)
            except:
                FirstFailure[i] = None
            Missing[i] = numRow - Observations[i]
            pVar[i] = 1 - self.varLevel[i]

        Excepted = Observations * pVar
        Ratio = Failures / Excepted

        table = PrettyTable()
        table.field_names = [
            "PorfolioID",
            "VarID",
            "VarLevel",
            "ObservationsLevel",
            "Observations",
            "Failures",
            "Excepted",
            "Ratio",
            "FirstFailure",
            "Missing",
        ]
        for i in range(numVar):
            row = [
                self.portfolioID,
                self.varID[i],
                self.varLevel[i],
                1 - pVar[i],
                Observations[i],
                Failures[i],
                Excepted[i],
                Ratio[i],
                FirstFailure[i],
                Missing[i],
            ]
            table.add_row(row)
        print(table)
        # return table

    def tbfi(self):
        numVar, numRow = len(self.varData), len(self.varData[0])

        tbfi = Tbfi()
        output = []

        table = PrettyTable()
        table.field_names = [
            "PortfolioID",
            "VaRID",
            "VaRLevel",
            "TBFI",
            "LRatioTBFI",
            "PValueTBFI",
            "Observations",
            "Failures",
            "TBFMin",
            "TBFQ1",
            "TBFQ2",
            "TBFQ3",
            "TBFMax",
            "TestLevel",
        ]
        for i in range(numVar):
            row = [self.portfolioID, self.varID[i], self.varLevel[i]]

            output = tbfi.test(self.failures[i], self.pVaR[i])

            tbf = output["dist"]
            tbfmin = tbf.min()
            tbfq1 = tbf[0]
            tbfq2 = tbf[1]
            tbfq3 = tbf[2]
            tbfmax = tbf.max()

            row.extend(
                [
                    output["result"],
                    output["LR"],
                    output["pVal"],
                    numRow,
                    sum(self.failures[i]),
                    tbfmin,
                    tbfq1,
                    tbfq2,
                    tbfq3,
                    tbfmax,
                    self.pVaR[i],
                ]
            )

            table.add_row(row)
        print(table)
