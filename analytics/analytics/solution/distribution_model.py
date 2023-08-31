from functools import reduce, cached_property, cache


class DistributionModel:
    def __init__(self, distribution=()):
        self._distribution = distribution

    @cache
    def mean(self):
        return reduce(lambda value, element: value + element[0]*element[1], enumerate(self._distribution), 0)

    @cache
    def variance(self):
        mean_square = self.mean()**2
        return reduce(lambda value, element: value + (element[0]**2)*element[1], enumerate(self._distribution), 0) - mean_square

    @property
    def distribution(self):
        return self._distribution



