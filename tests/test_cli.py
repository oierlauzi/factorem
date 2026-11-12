from __future__ import annotations

import subprocess
import sys

import jax
import pytest

import factorem
from factorem._device import resolve_device
from factorem.cli import build_parser


def test_arguments():
    arguments = build_parser().parse_args(
        ['in.star', 'out.star', '--diameter', '180']
    )
    assert arguments.input == 'in.star'
    assert arguments.output == 'out.star'
    assert arguments.diameter == 180.0
    assert arguments.resolution == 16.0
    assert arguments.components == 6
    assert arguments.device is None


@pytest.mark.parametrize(
    'arguments',
    [
        ['in.star', 'out.star'],
        ['in.star', 'out.star', '--diameter', '-1'],
        ['in.star', 'out.star', '--diameter', '100', '--components', '0'],
        ['in.star', 'out.star', '--diameter', '100', '--resolution', 'fine'],
    ],
)
def test_bad_arguments_are_refused(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)
    assert error.value.code == 2
    assert 'usage' in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(['--version'])
    assert error.value.code == 0
    assert factorem.__version__ in capsys.readouterr().out


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, '-m', 'factorem', '--help'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '--diameter' in result.stdout


def test_devices():
    assert resolve_device(None) in jax.devices()
    assert resolve_device('cpu').platform == 'cpu'
    device = jax.devices()[0]
    assert resolve_device(device) is device
    with pytest.raises(ValueError):
        resolve_device('gpu:99')
    with pytest.raises(ValueError):
        resolve_device('abacus')


def test_analysis_refuses_nonsense(simulation):
    particles = factorem.StarDataset.read(
        simulation.path, root=simulation.root
    ).particles
    with pytest.raises(ValueError):
        factorem.analyze(particles, diameter=-1.0)
    with pytest.raises(ValueError):
        factorem.analyze(particles, diameter=64.0, resolution=0.0)
    with pytest.raises(ValueError):
        factorem.analyze(particles, diameter=64.0, components=0)
