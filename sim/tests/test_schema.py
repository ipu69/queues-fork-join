from pydantic import TypeAdapter
from decimal import Decimal
from ..sim.schema import InputModel, LinearServicesInputModel, \
    FractionServiceInputModel


def test_implicit_input_model_deserialization__linear():
    inp_json = """{
        "input_type": "linear",
        "services": {
            "cv": 0.5,
            "skew": 14.0,
            "rate_min": 0.2,
            "rate_max": 0.8
        },
        "arrival": {
            "cv": 1.5,
            "skew": 0.5,
            "rate": 1.1,
            "lag": -0.05            
        },
        "num_servers": 7,
        "capacity": 42
    }"""
    
    adapter = TypeAdapter(InputModel)
    obj = adapter.validate_json(inp_json)
    
    assert obj is not None
    assert isinstance(obj, LinearServicesInputModel)
    assert obj.input_type == "linear"
    assert obj.services.cv == Decimal('0.5')
    assert obj.services.skew == Decimal('14.0')
    assert obj.services.rate_min == Decimal('0.2')
    assert obj.services.rate_max == Decimal('0.8')
    assert obj.arrival.cv == Decimal('1.5')
    assert obj.arrival.skew == Decimal('0.5')
    assert obj.arrival.rate == Decimal('1.1')
    assert obj.arrival.lag == Decimal('-0.05')
    assert obj.num_servers == 7
    assert obj.capacity == 42


def test_implicit_input_model_deserialization__fraction():
    inp_json = """{
        "input_type": "fraction",
        "services": {
            "cv": 0.9,
            "skew": 42.0,
            "rate_min": 0.3,
            "rate_max": 0.7
        },
        "arrival": {
            "cv": 1.6,
            "skew": 0.4,
            "rate": 1.2,
            "lag": -0.03            
        },
        "num_servers": 8,
        "num_slow_servers": 2,
        "capacity": 100
    }"""
    
    adapter = TypeAdapter(InputModel)
    obj = adapter.validate_json(inp_json)
    
    assert obj is not None
    assert isinstance(obj, FractionServiceInputModel)
    assert obj.input_type == "fraction"
    assert obj.services.cv == Decimal('0.9')
    assert obj.services.skew == Decimal('42.0')
    assert obj.services.rate_min == Decimal('0.3')
    assert obj.services.rate_max == Decimal('0.7')
    assert obj.arrival.cv == Decimal('1.6')
    assert obj.arrival.skew == Decimal('0.4')
    assert obj.arrival.rate == Decimal('1.2')
    assert obj.arrival.lag == Decimal('-0.03')
    assert obj.num_servers == 8
    assert obj.num_slow_servers == 2
    assert obj.capacity == 100
