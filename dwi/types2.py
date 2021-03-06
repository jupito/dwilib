"""Types."""

import dataclasses


class ConvertibleDataclass:
    """Data class with member functions for convenient conversion."""
    _astuple = dataclasses.astuple
    _asdict = dataclasses.asdict


class IterableDataclass:
    """Iterable data class."""
    def __iter__(self):
        return iter(dataclasses.astuple(self))


class SplitItemDataclass:
    """A data class that provides a way to format items using a separator."""
    _sep = '-'  # Item separator.

    @classmethod
    def parse(cls, s):
        """Parse a string of items separated by a separator."""
        fields = dataclasses.fields(cls)
        items = s.split(cls._sep)
        if len(items) > len(fields):
            raise ValueError(f'Too many items for {cls.__qualname__}: {s}')
        return cls(**{x.name: x.type(y) for x, y in zip(fields, items)})

    def __str__(self):
        """Format items separated by a separator."""
        items = (str(x) for x in dataclasses.astuple(self) if x is not None)
        return self._sep.join(items)


@dataclasses.dataclass(frozen=True)
class ImageMode(ConvertibleDataclass, SplitItemDataclass):
    """Image mode, e.g. `DWI-Mono-ADCm`."""
    modality: str
    model: str
    param: str


@dataclasses.dataclass(order=True, frozen=True)
class ImageTarget(ConvertibleDataclass, SplitItemDataclass):
    """Image target, e.g. `42-1a-1`."""
    case: int
    scan: str
    lesion: int


@dataclasses.dataclass(order=True, frozen=True)
class TextureSpec(ConvertibleDataclass, SplitItemDataclass):
    """Texture feature specification, e.g. `gabor-9-1,0.3,mean`."""
    method: str
    winsize: str  # Note that winsize can be an integer, `all`, or `mbb`.
    feature: str


@dataclasses.dataclass(order=True, frozen=True)
class GleasonScore(ConvertibleDataclass, IterableDataclass,
                   SplitItemDataclass):
    """Gleason score."""
    STANDARD_THRESHOLDS = '3+3', '3+4'
    primary: int
    secondary: int
    tertiary: int = None
    _sep = '+'

    def label(self, *threshold_seq):
        """Return label, when threshold is the maximum of the first class."""
        return self > self.__class__(*threshold_seq)
