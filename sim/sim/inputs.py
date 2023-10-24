import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

import numpy as np
from tqdm import tqdm

from pyqumo import Distribution, PhaseType, get_noncentral_m2, \
    get_noncentral_m3, BoundsError, MarkovArrival, HyperErlang, MatrixError
from pyqumo.algorithms.fitting import fit_acph2, fit_mern2, fit_map_horvath05, \
    optimize_lag1

from sim import schema

MAX_ORDER_FOR_LAGS = 20


@dataclass
class SimulateArgs:
    arrival: Distribution
    services: list[Distribution]
    capacities: list[int | None]
    max_packets: int = 100_000
    

def build_input(
    source: schema.InputModel | schema.ExplicitInputModel,
    max_packets: int = 100_000
) -> tuple[SimulateArgs, schema.FittedProps]:
    """
    Build named dictionary of arguments for pyqumo.simulate_forkjoin() call.
    """
    model = source.to_explicit()

    # Firstly, check that all services share the same cv and skew. If so, 
    # fit PH only once for all of them, and then just scale to the given rate.
    single_service_ph = True
    for service in model.services[1:]:
        if (service.cv != model.services[0].cv or 
                service.skew != model.services[0].skew):
            single_service_ph = False
            break
    if single_service_ph:
        base_ph = _fit_ph(
            cv=model.services[0].cv,
            skew=model.services[0].skew)
        ph_list = [
            base_ph.scale(1/float(service.rate)) 
            for service in model.services
        ]
    else:
        ph_list = [
            _fit_ph(cv=service.cv, skew=service.skew, mean=1/service.rate)
            for service in model.services
        ]
    
    # Arrival is always single, so fit it
    arrival_map = _fit_map(
        cv=model.arrival.cv,
        skew=model.arrival.skew,
        rate=model.arrival.rate,
        lag=model.arrival.lag
    )

    fitted_props = schema.FittedProps(
        arrival_cv=arrival_map.cv,
        arrival_skew=arrival_map.skewness,
        arrival_lag=arrival_map.lag(1),
        arrival_order=arrival_map.order,
    )

    if not isinstance(source, schema.ExplicitInputModel):
        fitted_props.service_cv = ph_list[0].cv
        fitted_props.service_skew = ph_list[0].skewness
        fitted_props.service_order = ph_list[0].order
    
    return SimulateArgs(
        arrival=arrival_map,
        services=ph_list,
        capacities=([model.capacity]*len(ph_list)),
        max_packets=max_packets,
    ), fitted_props


def generate_random_implicit_inputs(
        params: schema.RandomInputBounds,
        input_type: schema.InputType,
        num: int = 1,
        show_progress: bool = False
) -> list[schema.InputModel]:
    """
    Generate random implicit inputs.
    """

    def to_decimal(x: float):
        return Decimal(f"{x:.2f}")

    def random_int(bounds: tuple[int, int] | int) -> int:
        if isinstance(bounds, int):
            return bounds
        return random.randint(bounds[0], bounds[1])

    def random_float(bounds: tuple[float, float] | float) -> Decimal:
        if isinstance(bounds, float):
            return to_decimal(bounds)
        return to_decimal(random.random() * (bounds[1] - bounds[0]) + bounds[0])

    def random_skew(cv: Decimal, max_skew: float | None) -> Decimal:
        cv_f = float(cv)
        if cv_f <= 1.001:
            if 0.999 <= cv_f <= 1.001:
                min_skew = 2
            else:
                min_skew = 1/cv_f * (cv_f - 1/cv_f)
        else:
            min_skew = 0.5 * (3 * cv_f + 1 / cv_f**3)

        if max_skew is not None and max_skew > min_skew:
            return random_float((min_skew, max_skew))
        else:
            return to_decimal(min_skew)

    ans = []
    the_range = range(num)
    if show_progress:
        the_range = tqdm(the_range)

    for _ in the_range:
        # --------------------------
        # (1) Generate arrival props
        # --------------------------
        arrival_rate = random_float(params.arrival_rate)
        arrival_cv = random_float(params.arrival_cv)
        arrival_skew = random_skew(arrival_cv, params.arrival_skew_max)

        # Find minimum and maximum lags depending on the given CV and skew.
        arrival_ph = _fit_ph(cv=arrival_cv, skew=arrival_skew)
        if params.arrival_lag is not None:
            arrival_lag = random_float(params.arrival_lag)
        else:
            min_lag = optimize_lag1(arrival_ph, 'min').fun
            max_lag = optimize_lag1(arrival_ph, 'max').fun
            arrival_lag = random_float((min_lag, max_lag))
        arrival_map = _fit_map(
            arrival_rate, arrival_cv, arrival_skew, arrival_lag)

        # --------------------------
        # (2) Generate service props
        # --------------------------
        num_servers = random_int(params.num_servers)
        service_rate_min = random_float(params.service_rate)
        if num_servers > 1:
            service_rate_max = random_float(params.service_rate)
            if service_rate_min > service_rate_max:
                service_rate_min, service_rate_max = \
                    service_rate_max, service_rate_min
        else:
            service_rate_max = service_rate_min
        service_cv = random_float(params.service_cv)
        service_skew = random_skew(service_cv, params.service_skew_max)

        # ---------------------------
        # (3) Generate the rest props
        # ---------------------------
        capacity = random_int(params.capacity)

        model_args = dict(
            services=schema.ImplicitDistributionModel(
                rate_min=service_rate_min,
                rate_max=service_rate_max,
                cv=service_cv,
                skew=service_skew,
            ),
            arrival=schema.ArrivalModel(
                rate=to_decimal(arrival_map.rate),
                cv=to_decimal(arrival_map.cv),
                skew=to_decimal(arrival_map.skewness),
                lag=to_decimal(arrival_map.lag(1)),
            ),
            num_servers=num_servers,
            capacity=capacity,
        )

        if input_type is schema.InputType.FRACTION:
            if params.num_slow_servers is None:
                min_slow = 0
                max_slow = num_servers
            elif isinstance(params.num_slow_servers, int):
                min_slow = max_slow = min(params.num_slow_servers, num_servers)
            else:
                assert isinstance(params.num_slow_servers, Iterable)
                min_slow = min(num_servers, min(params.num_slow_servers))
                max_slow = min(num_servers, max(params.num_slow_servers))
            num_slow_servers = random_int((min_slow, max_slow))

            ans.append(schema.FractionServiceInputModel(
                input_type=schema.InputType.FRACTION,
                num_slow_servers=num_slow_servers,
                **model_args
            ))
        else:
            ans.append(schema.LinearServicesInputModel(
                input_type=schema.InputType.LINEAR,
                **model_args
            ))
    return ans


def _fit_map(rate: Decimal, cv: Decimal, skew: Decimal, lag: Decimal):
    ph = _fit_ph(cv=cv, skew=skew, mean=1/rate)
    if ph.order > MAX_ORDER_FOR_LAGS:
        return MarkovArrival.phase_type(ph.s, ph.p)
    it, max_it = 0, 10
    while True:
        try:
            arrival = fit_map_horvath05(ph, float(lag), n_iters=3, tol=0.1)[0]
        except MatrixError as er:
            # For some reason, generated MAP is not good. Try
            # another one.
            it += 1
            if it >= max_it:
                raise er
        else:
            return arrival.scale(ph.mean / arrival.mean)


def _fit_ph(cv: Decimal, skew: Decimal, mean: Decimal = 1.0,
            max_order: int | None = 20) -> PhaseType:
    """
    Fit PH-distribution by three moments using either ACPH(2) or MERN(2)
    method. See `pyqumo` documentation for details.

    Here mean value is assumed to be equal to 1.

    Parameters
    ----------
    cv : float
        coefficient of variation
    skew : float
        skewness value

    Returns
    -------
    ph : pyqumo.random.PhaseType
    """
    m1 = float(mean)
    m2 = get_noncentral_m2(m1, float(cv))
    m3 = get_noncentral_m3(m1, float(cv), float(skew))

    try:
        dist = fit_acph2([m1, m2, m3])[0]
    except BoundsError:
        dist = fit_mern2([m1, m2, m3], strict=False, max_shape_inc=1)[0]

        # If order is too large, we won't try to tune CV or skew.
        # Instead, we select a random shape and scale rate and shape.
        if max_order is not None and dist.order > max_order:
            bad_dist = dist  # store distribution
            new_shapes = np.random.randint(
                max_order // 4,
                max_order // 2 + 1, 2)
            k = new_shapes.astype(np.float32) / bad_dist.shapes
            dist = HyperErlang(
                shapes=new_shapes,
                params=(bad_dist.params * k),
                probs=bad_dist.probs)

            # Check the new distribution has good enough rate:
            if (err := abs(dist.rate - bad_dist.rate) / bad_dist.rate) > .01:
                print(f"!!! old rate was {bad_dist:.3f}, "
                      f"new rate is {dist.rate:.3f} [error = {err:.3f}]")
                assert False

    return dist.as_ph().scale(dist.mean / float(mean))
