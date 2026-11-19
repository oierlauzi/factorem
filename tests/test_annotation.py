from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import starfile
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2

from factorem.annotation import (
    COVARIANCES,
    MAX_CLASSES,
    AnnotationWindow,
    assign_to_spheres,
    fit_mixture,
    fit_mixture_by_bic,
    partition,
    sight,
    umap_projection,
)
from factorem.cli import build_parser, main
from factorem.io import (
    StarFormatError,
    format_embedding,
    read_embedding,
    write_classes,
)

_CENTRES = np.array([[-6.0, 0.0], [6.0, 0.0], [0.0, 9.0]])
_SIZES = (600, 400, 200)


def _blobs(dimensions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Three clouds of points far from each other, and the cloud of each
    point. They are apart in the first two dimensions only."""
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(3), _SIZES)
    points = rng.normal(size=(len(truth), dimensions))
    points[:, :2] += _CENTRES[truth]
    return points, truth


@pytest.fixture(autouse=True)
def _close_the_windows():
    yield
    plt.close('all')


def _mouse(
    window: AnnotationWindow,
    name: str,
    x: float,
    y: float,
    button: int = 1,
    step: int = 0,
) -> None:
    """What the mouse does at a point of the plane."""
    canvas = window.figure.canvas
    px, py = window._plot.transData.transform((x, y))
    event = MouseEvent(name, canvas, px, py, button=button, step=step)
    canvas.callbacks.process(name, event)


def _click(window: AnnotationWindow, x: float, y: float, button: int = 1):
    _mouse(window, 'button_press_event', x, y, button)
    _mouse(window, 'button_release_event', x, y, button)


def test_mixture_finds_the_clouds():
    points, truth = _blobs()
    found = fit_mixture(points, 3)
    # By decreasing size, which is the order of the clouds.
    np.testing.assert_array_equal(found.labels, truth)
    assert len(found.outlines) == 3
    for outline, centre in zip(found.outlines, _CENTRES, strict=True):
        np.testing.assert_allclose(outline.centre, centre, atol=0.3)
        # Two standard deviations to each side, of clouds of unit variance.
        assert 3.2 < outline.height <= outline.width < 4.8


def test_mixture_keeps_the_order_of_its_centres():
    points, truth = _blobs()
    found = fit_mixture(points, 3, centres=_CENTRES[::-1])
    np.testing.assert_array_equal(found.labels, 2 - truth)


def test_bic_counts_the_clouds():
    points, truth = _blobs()
    found = fit_mixture_by_bic(points, 6)
    assert len(found.outlines) == 3
    np.testing.assert_array_equal(found.labels, truth)
    # No more classes than points are tried.
    assert len(fit_mixture_by_bic(points[:2], 6).outlines) <= 2


@pytest.mark.parametrize('covariance', COVARIANCES)
def test_covariances(covariance):
    points, truth = _blobs()
    points[:, 0] *= 0.5
    found = fit_mixture(points, 3, covariance=covariance)
    np.testing.assert_array_equal(found.labels, truth)
    widths = np.array([outline.width for outline in found.outlines])
    heights = np.array([outline.height for outline in found.outlines])
    if covariance == 'spherical':
        np.testing.assert_allclose(widths, heights)
    else:
        # The clouds are half as wide as they are tall.
        np.testing.assert_allclose(heights / widths, 0.5, atol=0.1)
        angles = np.array([outline.angle for outline in found.outlines])
        np.testing.assert_allclose(np.abs(np.sin(np.radians(angles))), 1, 0.1)
    if covariance == 'tied':
        np.testing.assert_allclose(widths, widths[0])


def test_mixture_of_many_points_is_fitted_to_some(monkeypatch):
    points, truth = _blobs()
    monkeypatch.setattr(partition, '_FITTED_POINTS', 300)
    found = fit_mixture(points, 3)
    assert found.labels.shape == truth.shape
    np.testing.assert_array_equal(found.labels, truth)


def test_mixture_refuses_nonsense():
    points, _ = _blobs()
    with pytest.raises(ValueError):
        fit_mixture(points, 0)
    with pytest.raises(ValueError):
        fit_mixture(points, 3, centres=_CENTRES[:2])
    with pytest.raises(ValueError):
        fit_mixture(points[:2], 3)
    with pytest.raises(ValueError):
        fit_mixture_by_bic(points, 0)
    with pytest.raises(ValueError):
        fit_mixture_by_bic(points[:0], 3)


def test_spheres_take_the_closest_points():
    points, truth = _blobs()
    found = assign_to_spheres(points, _CENTRES, np.ones(3))
    np.testing.assert_array_equal(found.labels, truth)
    for outline, centre in zip(found.outlines, _CENTRES, strict=True):
        np.testing.assert_allclose(outline.centre, centre)
        assert outline.width == outline.height == 4.0


def test_a_wide_sphere_takes_what_is_far():
    centres = np.zeros((2, 2))
    points = np.array([[0.1, 0.0], [0.0, 5.0], [-9.0, 2.0]])
    found = assign_to_spheres(points, centres, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(found.labels, [0, 1, 1])


def test_spheres_refuse_nonsense():
    points, _ = _blobs()
    with pytest.raises(ValueError):
        assign_to_spheres(points, _CENTRES[:0], np.ones(0))
    with pytest.raises(ValueError):
        assign_to_spheres(points, _CENTRES, np.ones(2))
    with pytest.raises(ValueError):
        assign_to_spheres(points, _CENTRES, np.array([1.0, 0.0, 1.0]))


def test_sight_leaves_the_stragglers_out():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(10000, 2)) * (2.0, 1.0)
    points[0] = (500.0, -300.0)
    (left, right), (bottom, top) = sight(points, aspect=1.5)
    assert left < -2.0 < 2.0 < right < 3.0
    assert bottom < -1.0 < 1.0 < top < 2.5
    # No distance is distorted.
    np.testing.assert_allclose((right - left) / (top - bottom), 1.5)
    # A single point gets a view as well.
    (left, right), (bottom, top) = sight(np.ones((1, 2)))
    assert left < 1.0 < right and bottom < 1.0 < top


def test_umap_lays_the_clouds_apart():
    pytest.importorskip('umap')
    points, truth = _blobs(dimensions=4)
    plane = umap_projection(points)
    assert plane.shape == (len(points), 2)
    np.testing.assert_array_equal(fit_mixture(plane, 3).labels, truth)


def test_umap_asks_for_its_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, 'umap', None)
    with pytest.raises(ImportError, match=r'factorem\[umap\]'):
        umap_projection(np.zeros((10, 3)))
    # The window says so, and stays as it was.
    window = AnnotationWindow(_blobs(3)[0])
    window._attempt(window.show_umap)
    assert 'factorem[umap]' in window._status.get_text()
    assert window.points.shape == (1200, 2)


def _embedded_tables(coordinates: np.ndarray) -> dict[str, pd.DataFrame]:
    count = len(coordinates)
    optics = pd.DataFrame({'rlnOpticsGroup': [1], 'rlnVoltage': [300.0]})
    particles = pd.DataFrame(
        {
            'rlnImageName': [f'{i + 1}@stack.mrcs' for i in range(count)],
            'rlnOpticsGroup': np.ones(count, dtype=int),
            'factoremEmbedding': format_embedding(coordinates),
        }
    )
    return {'optics': optics, 'particles': particles}


def test_classes_round_trip(tmp_path):
    coordinates = np.random.default_rng(0).normal(size=(6, 3))
    coordinates[2] = np.nan
    source = tmp_path / 'embedded.star'
    starfile.write(_embedded_tables(coordinates), source)

    tables, read = read_embedding(source)
    np.testing.assert_allclose(read, coordinates, rtol=1e-4)
    target = tmp_path / 'annotated.star'
    write_classes(target, tables, np.arange(6))
    written = starfile.read(target, always_dict=True)
    assert list(written) == ['optics', 'particles']
    np.testing.assert_array_equal(
        written['particles']['factoremClass'], np.arange(6)
    )
    np.testing.assert_allclose(read_embedding(target)[1], coordinates, 1e-4)
    # What was read is left as it was.
    assert 'factoremClass' not in tables['particles']

    target = tmp_path / 'relion.star'
    write_classes(target, tables, np.arange(6), label='rlnClassNumber')
    written = starfile.read(target, always_dict=True)
    np.testing.assert_array_equal(
        written['particles']['rlnClassNumber'], np.arange(6)
    )
    with pytest.raises(ValueError):
        write_classes(target, tables, np.arange(5))


def test_embedding_must_be_there(tmp_path):
    tables = _embedded_tables(np.zeros((3, 2)))
    path = tmp_path / 'particles.star'
    starfile.write(
        {'particles': tables['particles'].drop(columns='factoremEmbedding')},
        path,
    )
    with pytest.raises(StarFormatError, match='factorem analyze'):
        read_embedding(path)

    tables['particles']['factoremEmbedding'] = ['[1, 2]', '[1]', '[x, 2]']
    path = tmp_path / 'ragged.star'
    starfile.write(tables, path)
    with pytest.raises(StarFormatError, match='cannot be read'):
        read_embedding(path)


def test_window_starts_with_a_single_class():
    points, _ = _blobs(dimensions=3)
    points[5] = np.nan
    points[7, 1] = np.inf
    window = AnnotationWindow(points)
    assert window.points.shape == (len(points) - 2, 2)
    expected = np.ones(len(points), dtype=int)
    expected[[5, 7]] = 0
    np.testing.assert_array_equal(window.classes, expected)
    assert window.partition is None and not window.saved


def test_window_refuses_what_has_no_plane():
    with pytest.raises(ValueError, match='two coordinates'):
        AnnotationWindow(np.zeros((10, 1)))
    with pytest.raises(ValueError, match='No particle'):
        AnnotationWindow(np.full((10, 3), np.nan))


def test_window_fits_the_classes():
    points, truth = _blobs(dimensions=3)
    points = np.concatenate((points, np.full((4, 3), np.nan)))
    truth = np.concatenate((truth + 1, np.zeros(4, dtype=int)))
    window = AnnotationWindow(points)

    window.fit(3)
    np.testing.assert_array_equal(window.classes, truth)
    window.clear_classes()
    assert set(window.classes) == {0, 1}

    window.fit_by_bic()
    np.testing.assert_array_equal(window.classes, truth)
    assert window._class_slider.val == 3
    assert 'BIC' in window._status.get_text()
    # The list of the classes: the three of them, and the particles of none.
    assert window._legend is not None
    listed = [text.get_text() for text in window._legend.get_texts()]
    assert listed == ['1: 600', '2: 400', '3: 200', '0: 4']

    window.covariance = 'spherical'
    window.fit(MAX_CLASSES)
    assert len(np.unique(window.classes)) == MAX_CLASSES + 1
    window.figure.canvas.draw()


def test_window_buttons_do_what_they_say():
    points, truth = _blobs()
    window = AnnotationWindow(points)
    labels = [button.label.get_text() for button in window._buttons]
    assert labels.index('Fit K classes') == 1
    window._class_slider.set_val(3)
    canvas = window.figure.canvas
    canvas.draw()
    for name in ('button_press_event', 'button_release_event'):
        x, y = window._buttons[1].ax.transAxes.transform((0.5, 0.5))
        event = MouseEvent(name, canvas, x, y, button=1)
        canvas.callbacks.process(name, event)
    np.testing.assert_array_equal(window.classes, truth + 1)

    window._covariance_radio.set_active(COVARIANCES.index('tied'))
    assert window.covariance == 'tied'


def test_window_sorts_by_its_seeds():
    points, truth = _blobs()
    window = AnnotationWindow(points)
    window.figure.canvas.draw()

    # Without seeds there is nothing to go by, and the window says so.
    window._attempt(window.take_seeds)
    assert window.partition is None
    assert 'seed' in window._status.get_text()

    # In the order of the smallest cloud to the largest.
    for x, y in _CENTRES[::-1]:
        _click(window, x, y)
    assert len(window.seeds) == 3
    window.take_seeds()
    np.testing.assert_array_equal(window.classes, 3 - truth)
    window.fit_seeds()
    np.testing.assert_array_equal(window.classes, 3 - truth)

    # A click on a seed adds none: it is dragged, resized or removed.
    x, y, deviation = window.seeds[0]
    _mouse(window, 'button_press_event', x, y)
    _mouse(window, 'motion_notify_event', x + 1.0, y - 2.0)
    _mouse(window, 'button_release_event', x + 1.0, y - 2.0)
    _mouse(window, 'motion_notify_event', x + 5.0, y - 5.0)
    assert len(window.seeds) == 3
    np.testing.assert_allclose(window.seeds[0], (x + 1.0, y - 2.0, deviation))
    _mouse(window, 'scroll_event', x + 1.0, y - 2.0, step=1)
    assert window.seeds[0][2] > deviation
    _mouse(window, 'scroll_event', x + 1.0, y - 2.0, step=-1)
    np.testing.assert_allclose(window.seeds[0][2], deviation)
    _click(window, x + 1.0, y - 2.0, button=3)
    assert len(window.seeds) == 2
    # A right click elsewhere does nothing.
    _click(window, 0.0, 0.0, button=3)
    assert len(window.seeds) == 2

    window.clear_seeds()
    assert window.seeds == []
    # The classes stay until they are cleared.
    assert window.partition is not None


def test_window_leaves_the_clicks_to_the_toolbar():
    window = AnnotationWindow(_blobs()[0])
    canvas = window.figure.canvas
    canvas.draw()

    toolbar = NavigationToolbar2(canvas)
    toolbar.zoom()
    _click(window, 0.0, 0.0)
    assert window.seeds == []
    toolbar.zoom()
    _click(window, 0.0, 0.0)
    assert len(window.seeds) == 1


def test_window_changes_of_plane():
    points, truth = _blobs(dimensions=4)
    # The clouds are apart in the last two coordinates instead.
    window = AnnotationWindow(points[:, ::-1])
    window.fit(3)
    window.add_seed(0.0, 0.0)

    window.show_components(3, 2)
    assert window.partition is None and window.seeds == []
    np.testing.assert_array_equal(window.points, points[:, :2])
    assert (window._x_slider.val, window._y_slider.val) == (4, 3)
    assert window._plot.get_xlabel() == 'Coordinate 4'
    window.fit(3)
    np.testing.assert_array_equal(window.classes, truth + 1)

    # A slider that is taken to the coordinate of the other swaps them.
    window._x_slider.set_val(3)
    assert (window._x_slider.val, window._y_slider.val) == (3, 4)
    np.testing.assert_array_equal(window.points, points[:, 1::-1])
    window._y_slider.set_val(1)
    np.testing.assert_array_equal(window.points, points[:, [1, 3]])

    with pytest.raises(ValueError):
        window.show_components(0, 0)
    with pytest.raises(ValueError):
        window.show_components(0, 4)


def test_window_counts_what_is_in_sight():
    points, _ = _blobs()
    window = AnnotationWindow(points)
    canvas = window.figure.canvas
    canvas.draw()
    whole = window._density.get_array()
    assert whole is not None and whole.sum() >= 0.99 * len(points)

    window._plot.set_xlim(-8.0, -4.0)
    window._plot.set_ylim(-1.5, 1.5)
    canvas.draw()
    zoomed = window._density.get_array()
    inside = (np.abs(points[:, 0] + 6.0) <= 2.0) & (np.abs(points[:, 1]) <= 1.5)
    assert zoomed is not None and zoomed.sum() == np.count_nonzero(inside)
    assert window._density.get_extent() == [-8.0, -4.0, -1.5, 1.5]
    # A seed is as large as the view asks for.
    window.add_seed(-6.0, 0.0)
    np.testing.assert_allclose(window.seeds[0][2], 0.04 * 3.0)


def test_window_saves_when_asked():
    points, truth = _blobs()
    saved = []
    window = AnnotationWindow(points, save=saved.append)
    window.fit(3)
    with pytest.raises(RuntimeError, match='display'):
        window.run()
    assert not saved and not window.saved
    window.save()
    assert window.saved
    np.testing.assert_array_equal(saved[0], truth + 1)
    assert not plt.fignum_exists(window.figure.number)


def test_window_stays_open_if_it_cannot_save():
    def fail(classes: np.ndarray) -> None:
        raise PermissionError('The disk is not to be written')

    window = AnnotationWindow(_blobs()[0], save=fail)
    window.save()
    assert not window.saved
    assert 'Not saved' in window._status.get_text()
    assert plt.fignum_exists(window.figure.number)


def test_annotate_arguments():
    arguments = build_parser().parse_args(['annotate', 'in.star', 'out.star'])
    assert arguments.command == 'annotate'
    assert arguments.input == 'in.star'
    assert arguments.output == 'out.star'
    assert arguments.label == 'factoremClass'
    assert not arguments.verbose

    arguments = build_parser().parse_args(
        ['annotate', 'in.star', 'out.star', '--label', 'rlnClassNumber', '-v']
    )
    assert arguments.label == 'rlnClassNumber'
    assert arguments.verbose

    with pytest.raises(SystemExit):
        build_parser().parse_args(['annotate', 'in.star'])


def test_annotate_command(tmp_path, monkeypatch, capsys):
    points, truth = _blobs(dimensions=3)
    points[0] = np.nan
    source = tmp_path / 'embedded.star'
    target = tmp_path / 'annotated.star'
    starfile.write(_embedded_tables(points), source)

    # There is no display to show the window on.
    assert main(['annotate', str(source), str(target)]) == 1
    assert 'display' in capsys.readouterr().err
    assert not target.exists()

    # The user closes the window.
    monkeypatch.setattr(AnnotationWindow, 'run', lambda self: None)
    assert main(['annotate', str(source), str(target)]) == 1
    assert 'without saving' in capsys.readouterr().err
    assert not target.exists()

    def run(self: AnnotationWindow) -> np.ndarray:
        self.fit(3)
        self.save()
        return self.classes

    monkeypatch.setattr(AnnotationWindow, 'run', run)
    arguments = [
        'annotate',
        str(source),
        str(target),
        '--label',
        'rlnClassNumber',
    ]
    assert main(arguments) == 0
    written = starfile.read(target, always_dict=True)['particles']
    expected = truth + 1
    expected[0] = 0
    np.testing.assert_array_equal(written['rlnClassNumber'], expected)
    assert 'factoremEmbedding' in written

    assert main(['annotate', str(tmp_path / 'none.star'), str(target)]) == 1
    assert 'factorem: error' in capsys.readouterr().err


def test_annotate_asks_for_its_extra(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'factorem.annotation', None)
    assert main(['annotate', 'in.star', 'out.star']) == 1
    assert 'factorem[annotate]' in capsys.readouterr().err
