from typing import Annotated, Literal
from pydantic import BaseModel, Field
from decimal import Decimal


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
    service: ImplicitDistributionModel
    arrival: ArrivalModel
    num_servers: int
    capacity: int


class LinearServicesInputModel(BaseImplicitModel):
    input_type: Literal['linear']
    

class FractionServiceInputModel(BaseImplicitModel):
    input_type: Literal['fraction']
    num_slow_servers: int

    
InputModel = Annotated[
    LinearServicesInputModel | FractionServiceInputModel, 
    Field(discriminator='input_type')
]


class ExplicitInputModel(BaseModel):
    arrival: ArrivalModel
    services: list[ExplicitDistributionModel]
    capacity: int


class StatsModel(BaseModel):
    avg: float
    std: float


class OutputModel(BaseModel):
    max_system_size: StatsModel
    min_system_size: StatsModel
    max_busy_rate: StatsModel
    min_busy_rate: StatsModel
    loss_prob: float
    response_time: StatsModel


class IOModel(BaseModel):
    inp: InputModel
    out: OutputModel


class RandomInputBounds(BaseModel):
    num_servers: tuple[int, int] = (1, 100)
    arrival_rate: tuple[float, float] = (0.1, 1.0)
    arrival_cv: tuple[float, float] = (0.1, 10.0)
    arrival_skew_max: float = 100.0
    arrival_lag: tuple[float, float] = (-0.1, 0.3)
    service_rate: tuple[float, float] = (0.1, 100.0)
    service_cv: tuple[float, float] = (0.1, 10.0)
    service_skew_max: float = 100.0
    capacity: tuple[int, int] = (0, 1000)
