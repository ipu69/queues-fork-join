import os
from multiprocessing import Pool
from pathlib import PurePath

import click
from tabulate import tabulate
from pydantic import TypeAdapter
import json
from tqdm import tqdm

from pyqumo.simulations.forkjoin.sandbox.model import simulate_forkjoin
import numpy as np

import schema
import inputs


@click.group()
def cli():
    pass


AnyInputModel = schema.InputModel | schema.ExplicitInputModel


def _run_simulation(
        args: tuple[
              list[AnyInputModel] | AnyInputModel,  # input model
              int,  # max_packets
              int,  # num_workers
        ]
):
    inp, max_packets, num_workers = args
    if isinstance(inp, list):
        all_inputs = [
            (inp_item, max_packets, num_workers)
            for inp_item in inp
        ]
        output_data = []
        with Pool(processes=num_workers) as pool:
            iterable = tqdm(pool.imap(_run_simulation, all_inputs),
                            total=len(inp))
            for result in iterable:
                if result is not None:
                    output_data.append(result)
        return output_data

    sim_args, fitted_props = inputs.build_input(inp, max_packets=max_packets)
    try:
        result = simulate_forkjoin(
            arrival=sim_args.arrival,
            services=sim_args.services,
            capacities=sim_args.capacities,
            max_packets=sim_args.max_packets
        )
    except Exception as ex:
        print(f"[ERROR] failed to run simulation for args: "
              f"{inp.__str__()}")
        return None

    out = schema.OutputModel.from_sim_results(result)
    io_model = schema.IOModel(
        inp=inp,
        out=out,
        meta=schema.Metadata(fitted=fitted_props)
    )
    return io_model


@cli.command
@click.option('-o', '--out', type=click.Path(), default=None)
@click.option("--pretty", is_flag=True, default=False,
              help="Write indented JSON")
@click.option('-j', '--num-workers', type=int, default=0,
              help="Number of workers, by default - number of available cores")
@click.option('-n', '--num-packets', type=int, default=500_000,
              help="Number of packets to simulate", show_default=True)
@click.argument('input_file', nargs=-1, type=click.Path(exists=True))
def simulate(input_file, num_packets, num_workers, pretty, out):
    # (1) Load data from input files
    inp_json = []
    for file_name in input_file:
        with open(file_name) as f:
            file_json = json.load(f)
            if not isinstance(file_json, list):
                inp_json.append(file_json)
            else:
                inp_json.extend(file_json) 

    # (2) Build input models from the JSON read
    inp_models = [
        TypeAdapter(schema.InputModel).validate_python(json_data)
        for json_data in inp_json
    ]

    # (3) Configure and run simulations
    if num_workers <= 0:
        num_workers = os.cpu_count()
    io_models = _run_simulation((inp_models, num_packets, num_workers))

    # (4) Build and print/write results
    io_models_list_model = schema.IOModelsList(io_models)
    dump_kwargs = {}
    if pretty:
        dump_kwargs.update({'indent': 2})
    answer = io_models_list_model.model_dump_json(**dump_kwargs)
    if out is None:
        print(answer)
    else:
        with open(out, 'w') as f:
            f.write(answer)


@cli.command
@click.option('-o', '--out', type=click.Path(), default=None)
@click.option('-t', '--type', 'input_type', help="Type of inputs",
              type=click.Choice([schema.InputType.LINEAR,
                                 schema.InputType.FRACTION]),
              default=schema.InputType.LINEAR.value, show_default=True)
@click.option("--pretty", is_flag=True, default=False,
              help="Write indented JSON")
@click.option("-P", "--stdout", is_flag=True, default=False,
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
