import numpy as np

class ParameterSet:
    def __init__(self, TimeCangeRate, CellSize, CellSizeNow, ParaArray):
        self.time = 1
        self.TimeCangeRate = TimeCangeRate
        self.xlim = CellSize[0]
        self.ylim = CellSize[1]
        self.xnow = CellSizeNow[0]
        self.ynow = CellSizeNow[1]
        self.Production_Prob = ParaArray[0]
        self.Degradation_Prob = ParaArray[1]
        self.Binding_Prob = ParaArray[2]
        self.Diffusion_Prob = ParaArray[3]

    def production_prob(self):
        # Even if the cell is growing, the production prob. is constant depending on gene exp.
        xchange = self.xnow/self.xlim
        ychange = self.ynow/self.ylim
        growth = xchange*ychange    # The changes of size caused by cell growth...

        # Parameters would change following the time scale.
        TimeScale = self.time
        prod_prob = self.Production_Prob*TimeScale/growth

        return prod_prob

    def degradation_prob(self):
        # deg_prob depends on time scale
        # Parameters would change following the time scale.
        TimeScale = self.time
        deg_prob = self.Degradation_Prob*TimeScale

        return deg_prob

    def binding_prob(self):
        # Parameters would change following the time scale.
        TimeScale = self.time
        bind_prob = self.Binding_Prob * TimeScale

        return bind_prob

    def diffusion_prob(self):
        # Parameters would change following the time scale.
        TimeScale = self.time
        diff_prob = self.Diffusion_Prob * TimeScale

        return diff_prob

    def time_changes(self):
        """
        Utilizing difference equation to change the time scale molecules are the way
        to avoid missing some details.
        """
        OriTimeScale = self.time
        rate = self.TimeCangeRate

        # Low down the computational cost when the time scale is a constant number
        if rate == 0:
            NewTimeScale = OriTimeScale
        else:
            NewTimeScale = OriTimeScale + rate

        self.time = NewTimeScale

        return NewTimeScale
