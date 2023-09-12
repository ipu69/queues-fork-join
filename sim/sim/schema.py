import enum
from typing import Annotated, Literal
from pydantic import BaseModel, Field, RootModel
from decimal import Decimal

from pyqumo.simulations.forkjoin.contract import ForkJoinResults


class InputType(str, enum.Enum):
    LINEAR = 'linear'
    FRACTION = 'fraction'


class BaseDistributionModel(BaseModel):
    cv: Decimal
    skew: Decimal


class ImplicitDistributionModel(BaseDistributionModel):
    rate_min: Decimal
    rate_max: Decimal


class ExplicitDistributionModel(BaseDistributionModel):
    rate: Decimal


class ArrivalModel(ExplicitDistributionModel):
    lag: Decimal


class BaseImplicitModel(BaseModel):
    services: ImplicitDistributionModel
    arrival: ArrivalModel
    num_servers: int
    capacity: int


class LinearServicesInputModel(BaseImplicitModel):
    input_type: Literal[InputType.LINEAR]
    
    def to_explicit(self):
        if self.num_servers <= 0:
            raise ValueError("Expected num_servers > 0")
        elif self.num_servers == 1:
            if self.services.rate_min != self.services.rate_max:
                raise ValueError("num_servers = 1 and rate_min != rate_max")
            else:
                return ExplicitInputModel(
                    arrival=self.arrival.model_copy(),
                    services=[ExplicitDistributionModel(
                        cv=self.services.cv,
                        skew=self.services.skew,
                        rate=self.services.rate_max
                    )],
                    capacity=self.capacity
                )
        
        k = ((self.services.rate_max - self.services.rate_min) / 
            (self.num_servers - 1))
        
        explicit_services = [
            ExplicitDistributionModel(
                cv=self.services.cv,
                skew=self.services.skew,
                rate=(self.services.rate_min + k*i)
            ) for i in range(self.num_servers)
        ]
        return ExplicitInputModel(
            arrival=self.arrival.model_copy(),
            services=explicit_services,
            capacity=self.capacity
        )


class FractionServiceInputModel(BaseImplicitModel):
    input_type: Literal[InputType.FRACTION]
    num_slow_servers: int

    def to_explicit(self):
        if self.num_servers <= 0:
            raise ValueError("Expected num_servers > 0")
        if self.num_slow_servers > self.num_servers:
            raise ValueError("Expected num_slow_servers <= num_servers")
        
        slow_service_model = ExplicitInputModel(
            cv=self.services.cv,
            skew=self.services.skew,
            rate=self.services.rate_min
        )
        fast_service_model = ExplicitInputModel(
            cv=self.services.cv,
            skew=self.services.skew,
            rate=self.services.rate_max
        )
        
        slow_services = [
            slow_service_model.model_copy() 
            for _ in range(self.num_slow_servers)
        ]
        fast_services = [
            fast_service_model.model_copy() 
            for _ in range(self.num_servers - self.num_slow_servers)
        ]
        return ExplicitInputModel(
            arrival=self.arrival.model_copy(),
            services=(slow_services + fast_services),
            capacity=self.capacity
        )


InputModel = Annotated[
    LinearServicesInputModel | FractionServiceInputModel, 
    Field(discriminator='input_type')
]


InputModelList = RootModel[list[InputModel]]


class ExplicitInputModel(BaseModel):
    arrival: ArrivalModel
    services: list[ExplicitDistributionModel]
    capacity: int
    
    def to_explicit(self):
        return self


class StatsModel(BaseModel):
    avg: float
    std: float
    index: int | None = None


class OutputModel(BaseModel):
    max_system_size: StatsModel
    min_system_size: StatsModel
    max_busy_rate: StatsModel
    min_busy_rate: StatsModel
    loss_prob: float
    response_time: StatsModel

    @staticmethod
    def from_sim_results(results: ForkJoinResults):

        def find_minmax_stats(dist_list) -> tuple[StatsModel, StatsModel]:
            first = dist_list[0]
            max_stats = StatsModel(avg=first.mean, std=first.std, index=0)
            min_stats = StatsModel(avg=first.mean, std=first.std, index=0)
            for i, dist in enumerate(dist_list[1:]):
                if dist.mean > max_stats.avg:
                    max_stats.avg = dist.mean
                    max_stats.std = dist.std
                    max_stats.index = i+1
                elif dist.mean < min_stats.avg:
                    min_stats.avg = dist.mean
                    min_stats.std = dist.std
                    min_stats.index = i+1
            return min_stats, max_stats

        min_ss, max_ss = find_minmax_stats(results.system_sizes)
        min_busy, max_busy = find_minmax_stats(results.busy)

        return OutputModel(
            max_system_size=max_ss,
            min_system_size=min_ss,
            max_busy_rate=max_busy,
            min_busy_rate=min_busy,
            loss_prob=results.packet_drop_prob,
            response_time=StatsModel(
                avg=results.response_time.avg,
                std=results.response_time.std
            ),
        )


class FittedProps(BaseModel):
    arrival_cv: float
    arrival_skew: float
    arrival_lag: float
    arrival_order: int
    service_cv: float | None = None
    service_skew: float | None = None
    service_order: int | None = None


class Metadata(BaseModel):
    fitted: FittedProps


class IOModel(BaseModel):
    inp: InputModel
    meta: Metadata
    out: OutputModel


IOModelsList = RootModel[list[IOModel]]


class RandomInputBounds(BaseModel):
    num_servers: tuple[int, int] | int = (1, 100)
    arrival_rate: tuple[float, float] | float = (0.1, 1.0)
    arrival_cv: tuple[float, float] | float = (0.1, 10.0)
    arrival_skew_max: float | None = 100.0
    arrival_lag: tuple[float, float] | float | None = (-0.1, 0.3)
    service_rate: tuple[float, float] | float = (0.1, 100.0)
    service_cv: tuple[float, float] | float = (0.1, 10.0)
    service_skew_max: float = 100.0
    capacity: tuple[int, int] | int = (0, 1000)
    num_slow_servers: tuple[int, int] | int | None = None

