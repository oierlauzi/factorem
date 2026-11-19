"""The window that the particles are annotated in.

It shows the density of the particles on a plane, and sorts them into
classes there. The plane is that of two of their coordinates, or the one
UMAP lays all of the coordinates out on. The classes are the Gaussians of a
mixture, which is either fitted to the particles or placed by hand:

- With a number of classes, or with as many as the BIC asks for.
- From seeds, which a fit then refines.
- With the seeds as they are, a round Gaussian each.

A click on the plane adds a seed, dragging one moves it, the wheel over
one changes its size and a right click removes it. The zoom and the pan of
the toolbar take the clicks while they are on, and the density is counted
again for what they leave in sight.

The classes are numbered from one, and every particle on the plane is of
one and only one of them: of the first, until they are sorted. The
particles without coordinates are of class zero.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.colors import PowerNorm
from matplotlib.legend import Legend
from matplotlib.patches import Circle, Ellipse, Patch
from matplotlib.patheffects import withStroke
from matplotlib.widgets import Button, RadioButtons, Slider

from .partition import (
    COVARIANCES,
    OUTLINE_DEVIATIONS,
    Covariance,
    Partition,
    assign_to_spheres,
    fit_mixture,
    fit_mixture_by_bic,
)
from .projection import sight, umap_projection

logger = logging.getLogger(__name__)

MAX_CLASSES = 20
"""The most classes that the window fits by itself."""

# The colours of the classes: the ten of the palette first, and then their
# light versions.
_COLOURS = colormaps['tab20']([*range(0, 20, 2), *range(1, 20, 2)])

# Particles drawn in the colour of their class, at most. More of them
# would only make the window slow.
_DRAWN_POINTS = 50_000

# Standard deviation of a new seed over the height of the view, and what
# a step of the wheel multiplies it by.
_SEED_SHARE = 0.04
_SEED_STEP = 1.2

# How close to a seed a click has to be to mean it [pixels].
_PICK_RADIUS = 10.0

# Backends of matplotlib that draw to files, and so have no window.
_WINDOWLESS = ('agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template')

# Where the controls are, in fractions of the figure.
_LEFT, _WIDTH = 0.765, 0.22
_HALO = [withStroke(linewidth=3.0, foreground='white')]


def _bins(count: int) -> int:
    """Bins along the height of the view: fine enough to show the shape of
    the density, and coarse enough for it not to be all noise."""
    return int(np.clip(np.sqrt(count / 10.0), 32, 512))


class AnnotationWindow:
    """A window that shows particles on a plane and sorts them into
    classes.

    What its controls do, its methods do as well.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        *,
        save: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """
        Args:
            coordinates: The coordinates of each particle,
                ``(particles, components)``. The particles with something
                other than finite numbers are left out of the plane.
            save: What saves the class of each particle, as `classes`
                gives them, when asked to. The window stays open if it
                raises an `OSError`.

        Raises:
            ValueError: If there are fewer than two coordinates, or no
                particle has them.
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(
                'The particles need two coordinates at least, to be shown '
                'on a plane.'
            )
        self._located = np.isfinite(coordinates).all(axis=1)
        if not self._located.any():
            raise ValueError('No particle has coordinates.')
        self._coordinates = coordinates[self._located]
        self._save = save

        self.saved = False
        """Whether the classes were saved."""
        self.partition: Partition | None = None
        """The classes of the particles on the plane, once sorted."""
        self.seeds: list[tuple[float, float, float]] = []
        """The centre, as two coordinates of the plane, and the standard
        deviation of each seed."""
        self.covariance: Covariance = 'full'
        """What the covariances of the fitted Gaussians may be."""

        self._components: tuple[int, int] | None = (0, 1)
        self._umap: np.ndarray | None = None
        self._points: np.ndarray = self._coordinates[:, :2]
        self._dragged: int | None = None

        count = len(self._coordinates)
        self._drawn = np.arange(count)
        if count > _DRAWN_POINTS:
            self._drawn = np.random.default_rng(0).choice(
                count, _DRAWN_POINTS, replace=False
            )

        self._class_artists: list[Artist] = []
        self._seed_artists: list[Artist] = []
        self._legend: Legend | None = None
        self._binned: tuple[float, float, float, float] | None = None
        self._build()
        self._lay_out()

    # What the window holds.

    @property
    def points(self) -> np.ndarray:
        """Where each particle with coordinates is on the plane,
        ``(particles, 2)``."""
        return self._points

    @property
    def classes(self) -> np.ndarray:
        """The class of each particle, ``(particles,)``, counted from one.
        It is zero for the particles without coordinates."""
        classes = np.zeros(len(self._located), dtype=np.int64)
        classes[self._located] = (
            1 if self.partition is None else self.partition.labels + 1
        )
        return classes

    # The plane.

    def show_components(self, first: int, second: int) -> None:
        """Show the plane of two of the coordinates, counted from zero.
        The classes and the seeds, which are of the plane that was shown,
        are dropped."""
        count = self._coordinates.shape[1]
        if not (0 <= first < count and 0 <= second < count):
            raise ValueError(f'The particles have {count} coordinates.')
        if first == second:
            raise ValueError('A plane needs two different coordinates.')
        self._components = (first, second)
        self._points = self._coordinates[:, [first, second]]
        self._set_sliders(first, second)
        self._lay_out()
        self._report(f'Coordinates {first + 1} and {second + 1}.')

    def show_umap(self) -> None:
        """Show the plane that UMAP lays all of the coordinates out on.
        The classes and the seeds, which are of the plane that was shown,
        are dropped.

        Raises:
            ImportError: Without umap-learn.
        """
        if self._umap is None:
            self._report('Laying the particles out by UMAP...', now=True)
            try:
                self._umap = umap_projection(self._coordinates)
            except ImportError as error:
                self._report(str(error))
                raise
        self._components = None
        self._points = self._umap
        self._lay_out()
        self._report('UMAP of all the coordinates.')

    # The classes.

    def fit(self, classes: int) -> None:
        """Sort the particles by a mixture of that many Gaussians."""
        partition = fit_mixture(
            self._points, classes, covariance=self.covariance
        )
        self._sort(partition, f'Fitted {classes} classes.')

    def fit_by_bic(self) -> None:
        """Sort the particles by a mixture of as many Gaussians as its BIC
        asks for, up to `MAX_CLASSES`."""
        self._report('Fitting from 1 class on...', now=True)
        partition = fit_mixture_by_bic(
            self._points, MAX_CLASSES, covariance=self.covariance
        )
        count = len(partition.outlines)
        self._class_slider.set_val(count)
        self._sort(partition, f'The BIC asks for {count} classes.')

    def fit_seeds(self) -> None:
        """Sort the particles by a mixture that is fitted from the seeds,
        of a Gaussian each."""
        seeds = self._placed_seeds()
        partition = fit_mixture(
            self._points,
            len(seeds),
            covariance=self.covariance,
            centres=seeds[:, :2],
        )
        self._sort(partition, f'Fitted {len(seeds)} classes from the seeds.')

    def take_seeds(self) -> None:
        """Sort the particles by the seeds as they are, each a round
        Gaussian."""
        seeds = self._placed_seeds()
        partition = assign_to_spheres(self._points, seeds[:, :2], seeds[:, 2])
        self._sort(partition, f'Took the {len(seeds)} seeds as classes.')

    def clear_classes(self) -> None:
        """Take all the particles back to a single class."""
        self._sort(None, 'A single class.')

    def add_seed(self, x: float, y: float) -> None:
        """Place a seed on the plane, of a size that goes by how much of
        the plane is seen."""
        _, _, bottom, top = self._view()
        self.seeds.append((x, y, _SEED_SHARE * (top - bottom)))
        self._draw_seeds()
        self._report(f'{len(self.seeds)} seeds.')

    def clear_seeds(self) -> None:
        """Remove all the seeds."""
        self.seeds = []
        self._draw_seeds()
        self._report('No seeds.')

    def save(self) -> None:
        """Save the classes and close the window."""
        try:
            if self._save is not None:
                self._save(self.classes)
        except OSError as error:
            self._report(f'Not saved: {error}')
            return
        self.saved = True
        plt.close(self.figure)

    def run(self) -> np.ndarray | None:
        """Show the window until it is closed.

        Returns:
            The class of each particle, as `classes` gives them, or
            nothing if the window was closed without saving.

        Raises:
            RuntimeError: If there is no display to show the window on.
        """
        if matplotlib.get_backend().lower() in _WINDOWLESS:
            raise RuntimeError(
                'There is no display to show the window on: matplotlib '
                f'draws with {matplotlib.get_backend()}.'
            )
        plt.show()
        return self.classes if self.saved else None

    # The making of the window.

    def _build(self) -> None:
        self.figure = plt.figure(figsize=(13.0, 7.5))
        manager = self.figure.canvas.manager
        if manager is not None:
            manager.set_window_title('factorem annotate')
        self._plot = self.figure.add_axes((0.055, 0.09, 0.665, 0.86))

        def header(y: float, text: str) -> None:
            self.figure.text(_LEFT, y, text, fontsize=9, weight='bold')

        def slider(y: float, label: str, top: int, start: int) -> Slider:
            axes = self.figure.add_axes((_LEFT + 0.03, y, _WIDTH - 0.06, 0.02))
            return Slider(axes, label, 1, max(top, 2), valinit=start, valstep=1)

        def button(y: float, label: str, action: Callable[[], None]) -> Button:
            axes = self.figure.add_axes((_LEFT, y, _WIDTH, 0.036))
            widget = Button(axes, label)
            widget.on_clicked(lambda _event: self._attempt(action))
            return widget

        components = self._coordinates.shape[1]
        header(0.962, 'Plane')
        self._x_slider = slider(0.928, 'x', components, 1)
        self._y_slider = slider(0.898, 'y', components, 2)
        self._x_slider.on_changed(lambda _value: self._on_components(True))
        self._y_slider.on_changed(lambda _value: self._on_components(False))

        header(0.812, 'Classes')
        self._covariance_radio = RadioButtons(
            self.figure.add_axes((_LEFT, 0.712, _WIDTH, 0.095)),
            COVARIANCES,
            active=COVARIANCES.index(self.covariance),
        )
        self._covariance_radio.on_clicked(self._on_covariance)
        self._class_slider = slider(0.678, 'K', MAX_CLASSES, 3)

        header(0.548, 'Seeds')
        self.figure.text(
            _LEFT,
            0.508,
            'Click to add, drag to move, scroll\nto resize, right click to '
            'remove.',
            fontsize=7.5,
        )

        # Kept, for a widget that nothing refers to stops answering.
        self._buttons = [
            button(0.848, 'Lay out by UMAP', self.show_umap),
            button(
                0.63,
                'Fit K classes',
                lambda: self.fit(int(self._class_slider.val)),
            ),
            button(0.586, 'Fit as many as the BIC asks', self.fit_by_bic),
            button(0.458, 'Fit from the seeds', self.fit_seeds),
            button(0.414, 'Take the seeds as they are', self.take_seeds),
            button(0.37, 'Clear the seeds', self.clear_seeds),
            button(0.31, 'Clear the classes', self.clear_classes),
            button(0.266, 'Save and close', self.save),
        ]
        self._status = self.figure.text(
            _LEFT, 0.25, '', fontsize=8.5, va='top', wrap=True
        )

        canvas = self.figure.canvas
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        canvas.mpl_connect('scroll_event', self._on_scroll)
        canvas.mpl_connect('draw_event', self._on_draw)

    def _lay_out(self) -> None:
        """Draw the density of the particles on the plane that is shown.
        The classes and the seeds were of some other plane."""
        self.partition = None
        self.seeds = []
        self._dragged = None

        # Clearing the axes takes what was drawn on them along.
        self._plot.clear()
        self._class_artists = []
        self._seed_artists = []
        self._density = self._plot.imshow(
            np.ma.masked_equal(np.zeros((1, 1)), 0.0),
            origin='lower',
            cmap='Greys',
            interpolation='nearest',
            zorder=1,
        )
        self._binned = None

        # The view takes all the room there is for it. It is the room that
        # then follows the shape of what a zoom leaves in sight.
        room = self._plot.get_position(original=True)
        width, height = self.figure.get_size_inches()
        limits = sight(
            self._points, room.width * width / (room.height * height)
        )
        self._plot.set_xlim(*limits[0])
        self._plot.set_ylim(*limits[1])
        self._plot.set_aspect('equal', adjustable='box')
        self._bin()
        if self._components is None:
            self._plot.set_xlabel('UMAP 1')
            self._plot.set_ylabel('UMAP 2')
        else:
            first, second = self._components
            self._plot.set_xlabel(f'Coordinate {first + 1}')
            self._plot.set_ylabel(f'Coordinate {second + 1}')
        self._draw_classes()

    def _view(self) -> tuple[float, float, float, float]:
        """What of the plane is seen: left, right, bottom and top."""
        left, right = sorted(self._plot.get_xlim())
        bottom, top = sorted(self._plot.get_ylim())
        return (float(left), float(right), float(bottom), float(top))

    def _bin(self) -> bool:
        """Count the particles in bins of what is seen of the plane, so
        that the density stays sharp when the view is zoomed.

        Returns:
            Whether the view had changed since the last time.
        """
        view = self._view()
        if view == self._binned:
            return False
        left, right, bottom, top = view
        x, y = self._points.T
        seen = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)
        rows = _bins(int(np.count_nonzero(seen)))
        columns = max(round(rows * (right - left) / (top - bottom)), 1)
        counts, _, _ = np.histogram2d(
            x[seen],
            y[seen],
            bins=(columns, rows),
            range=((left, right), (bottom, top)),
        )
        self._density.set_data(np.ma.masked_equal(counts.T, 0.0))
        self._density.set_extent(view)
        # The square root makes the noise of the counts as strong at every
        # density.
        self._density.set_norm(
            PowerNorm(0.5, vmin=0.0, vmax=max(float(counts.max()), 1.0))
        )
        self._binned = view
        return True

    def _draw_classes(self) -> None:
        for artist in self._class_artists:
            artist.remove()
        self._class_artists = []

        if self.partition is not None:
            drawn = self._points[self._drawn]
            colours = _COLOURS[
                self.partition.labels[self._drawn] % len(_COLOURS)
            ]
            self._class_artists.append(
                self._plot.scatter(
                    drawn[:, 0],
                    drawn[:, 1],
                    c=colours,
                    s=4.0,
                    alpha=0.35,
                    linewidths=0.0,
                    zorder=2,
                )
            )
            for index, outline in enumerate(self.partition.outlines):
                centre = (float(outline.centre[0]), float(outline.centre[1]))
                ellipse = Ellipse(
                    centre,
                    outline.width,
                    outline.height,
                    angle=outline.angle,
                    fill=False,
                    edgecolor=_COLOURS[index % len(_COLOURS)],
                    linewidth=2.0,
                    zorder=4,
                )
                self._plot.add_patch(ellipse)
                # The number tells the classes apart where the colours of
                # two of them are alike.
                number = self._plot.annotate(
                    str(index + 1),
                    centre,
                    ha='center',
                    va='center',
                    fontsize=11,
                    weight='bold',
                    path_effects=_HALO,
                    zorder=7,
                    annotation_clip=True,
                )
                self._class_artists += [ellipse, number]
        self._draw_counts()
        self.figure.canvas.draw_idle()

    def _draw_counts(self) -> None:
        """List how many particles each class has."""
        if self._legend is not None:
            self._legend.remove()
        classes = self.classes
        count = 1 if self.partition is None else len(self.partition.outlines)
        counts = np.bincount(classes, minlength=count + 1)
        handles = [
            Patch(color=_COLOURS[index % len(_COLOURS)])
            for index in range(count)
        ]
        labels = [
            f'{index}: {counts[index]:,}' for index in range(1, count + 1)
        ]
        if counts[0]:
            handles.append(Patch(facecolor='none', edgecolor='0.5'))
            labels.append(f'0: {counts[0]:,}')
        self._legend = self.figure.legend(
            handles,
            labels,
            title='Particles of each class',
            loc='upper left',
            bbox_to_anchor=(_LEFT, 0.2),
            ncols=1 + (len(labels) - 1) // 8,
            fontsize=8,
            title_fontsize=8.5,
            alignment='left',
            frameon=False,
            borderpad=0.0,
            borderaxespad=0.0,
            labelspacing=0.25,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.8,
        )

    def _draw_seeds(self) -> None:
        for artist in self._seed_artists:
            artist.remove()
        self._seed_artists = []
        for x, y, deviation in self.seeds:
            # As large as the outline of the class it would be.
            circle = Circle(
                (x, y),
                OUTLINE_DEVIATIONS * deviation,
                fill=False,
                edgecolor='black',
                linestyle='--',
                linewidth=1.2,
                path_effects=_HALO,
                zorder=5,
            )
            self._plot.add_patch(circle)
            (cross,) = self._plot.plot(
                [x],
                [y],
                marker='x',
                color='black',
                markersize=9.0,
                markeredgewidth=2.0,
                path_effects=_HALO,
                zorder=6,
                scalex=False,
                scaley=False,
            )
            self._seed_artists += [circle, cross]
        self.figure.canvas.draw_idle()

    # What the controls do.

    def _sort(self, partition: Partition | None, message: str) -> None:
        self.partition = partition
        self._draw_classes()
        self._report(message)

    def _placed_seeds(self) -> np.ndarray:
        if not self.seeds:
            raise ValueError('Click on the plane to place a seed first.')
        return np.array(self.seeds, dtype=np.float64)

    def _report(self, message: str, now: bool = False) -> None:
        """Tell the user something, right away if what follows is slow."""
        logger.info('%s', message)
        self._status.set_text(message)
        if now:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        else:
            self.figure.canvas.draw_idle()

    def _attempt(self, action: Callable[[], None]) -> None:
        """Do something, or tell the user why it cannot be done."""
        try:
            action()
        except (ValueError, ImportError) as error:
            self._report(str(error))

    def _set_sliders(self, first: int, second: int) -> None:
        for slider, component in (
            (self._x_slider, first),
            (self._y_slider, second),
        ):
            # Without the slider answering, as if it had been moved.
            slider.eventson = False
            slider.set_val(component + 1)
            slider.eventson = True

    def _on_components(self, of_x: bool) -> None:
        first = int(self._x_slider.val) - 1
        second = int(self._y_slider.val) - 1
        if first == second and self._components is not None:
            # The axes swap rather than show the same coordinate twice.
            if of_x:
                second = self._components[0]
            else:
                first = self._components[1]
        self._attempt(lambda: self.show_components(first, second))

    def _on_draw(self, _event: Event) -> None:
        # The limits of the view are only settled when it is drawn.
        if self._bin():
            self.figure.canvas.draw_idle()

    def _on_covariance(self, label: str | None) -> None:
        for covariance in COVARIANCES:
            if covariance == label:
                self.covariance = covariance

    def _busy(self) -> bool:
        """Whether the clicks are for the zoom or the pan of the toolbar."""
        toolbar = self.figure.canvas.toolbar
        return toolbar is not None and bool(toolbar.mode)

    def _seed_at(self, event: MouseEvent) -> int | None:
        """The seed that a click means, by how far from it on the screen."""
        if not self.seeds:
            return None
        centres = np.array(self.seeds)[:, :2]
        pixels = self._plot.transData.transform(centres)
        distances = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        nearest = int(np.argmin(distances))
        return nearest if distances[nearest] <= _PICK_RADIUS else None

    def _on_press(self, event: Event) -> None:
        if (
            not isinstance(event, MouseEvent)
            or event.inaxes is not self._plot
            or event.xdata is None
            or event.ydata is None
            or self._busy()
        ):
            return
        index = self._seed_at(event)
        if event.button == 3 and index is not None:
            del self.seeds[index]
            self._draw_seeds()
            self._report(f'{len(self.seeds)} seeds.')
        elif event.button == 1 and index is not None:
            self._dragged = index
        elif event.button == 1:
            self.add_seed(event.xdata, event.ydata)
            self._dragged = len(self.seeds) - 1

    def _on_motion(self, event: Event) -> None:
        if (
            self._dragged is None
            or not isinstance(event, MouseEvent)
            or event.inaxes is not self._plot
            or event.xdata is None
            or event.ydata is None
        ):
            return
        deviation = self.seeds[self._dragged][2]
        self.seeds[self._dragged] = (event.xdata, event.ydata, deviation)
        self._draw_seeds()

    def _on_release(self, _event: Event) -> None:
        self._dragged = None

    def _on_scroll(self, event: Event) -> None:
        if (
            not isinstance(event, MouseEvent)
            or event.inaxes is not self._plot
            or self._busy()
        ):
            return
        index = self._seed_at(event)
        if index is None:
            return
        x, y, deviation = self.seeds[index]
        deviation *= _SEED_STEP if event.step > 0 else 1.0 / _SEED_STEP
        self.seeds[index] = (x, y, deviation)
        self._draw_seeds()
