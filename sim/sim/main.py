from pathlib import PurePath

import click
from tabulate import tabulate
from pydantic import TypeAdapter
import json

from pyqumo.simulations.forkjoin.sandbox.model import simulate_forkjoin
import numpy as np

import schema
import inputs


@click.group()
def cli():
    pass


@cli.command
def simulate():
    inp_json = """{
        "input_type": "linear",
        "services": {
            "cv": 0.5,
            "skew": 14.0,
            "rate_min": 0.2,
            "rate_max": 0.8
        },
        "arrival": {
            "cv": 0.8,
            "skew": 3,
            "rate": 1.0,
            "lag": 0.1
        },
        "num_servers": 7,
        "capacity": 42
    }"""
    inp = TypeAdapter(schema.InputModel).validate_json(inp_json)
    sim_args, fitted_props = inputs.build_input(inp, max_packets=100_000)
    results = simulate_forkjoin(
        arrival=sim_args.arrival,
        services=sim_args.services,
        capacities=sim_args.capacities,
        max_packets=sim_args.max_packets
    )
    out = schema.OutputModel.from_sim_results(results)
    io_model = schema.IOModel(
        inp=inp,
        out=out,
        meta=schema.Metadata(fitted=fitted_props)
    )
    print(json.dumps(io_model.model_dump(), indent=4, default=str))


@cli.command
@click.option('-o', '--out', type=click.Path(), default=None)
@click.option('-t', '--type',  'input_type', help="Type of inputs",
              type=click.Choice([schema.InputType.LINEAR,
                                 schema.InputType.FRACTION]),
              default=schema.InputType.LINEAR.value, show_default=True)
@click.option("--pretty", is_flag=True, default=False, help="Write indented JSON")
@click.option("-P", "--stdout", default=False,
              help="Rather writing to a file, print result to stdout")
@click.option("-S", "--silent", is_flag=True, default=False,
              help="Don't print progress or any additional runtime info")
@click.option("-n", "--number", default=100, show_default=True,
              help="Number of records to generate")
@click.argument('input_file', type=click.Path(exists=True))
def generate(input_file, number, silent, stdout, pretty, input_type, out):
    with open(input_file) as f:
        inp_json = json.load(f)

    params = schema.RandomInputBounds.model_validate(inp_json)

    gen_kwargs = {
        'params': params,
        'input_type': input_type,
        'num': number,
    }
    if not silent:
        gen_kwargs.update({'show_progress': True})

    inp_list = inputs.generate_random_implicit_inputs(**gen_kwargs)

    if not silent:
        lags_hist = np.histogram([float(inp.arrival.lag) for inp in inp_list])
        lags_hist_lines = []

        for i, (val, right) in enumerate(zip(lags_hist[0], lags_hist[1][1:])):
            left = lags_hist[1][i]
            lags_hist_lines.append((
                f"({left:.3f}, {right:.3f})",
                f"{float(val) / len(inp_list) * 100:.2f}% "
                f"({val}/{len(inp_list)})"
            ))
        print("\n*** STATISTICS ***\n")
        print(tabulate(lags_hist_lines, headers=["Lag-1 range", "Fraction"]))

    dump_kwargs = {}
    if pretty:
        dump_kwargs = {**dump_kwargs, 'indent': 2}
    out_json = schema.InputModelList(inp_list).model_dump_json(**dump_kwargs)

    if not stdout:
        if out is None:
            out = PurePath(input_file)
            suffix = "".join(out.suffixes)
            name = out.name[:-len(suffix)]
            out = out.with_name(name + "_data.json")

        with open(out, 'w') as f:
            f.write(out_json)
    else:
        print(out_json)


if __name__ == '__main__':
    cli()
