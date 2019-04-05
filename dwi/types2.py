"""Types."""

import dataclasses


class ConvertibleDataclass:
    """..."""
    _astuple = dataclasses.astuple
    _asdict = dataclasses.asdict


class IterableDataclass:
    """..."""
    def __iter__(self):
        return iter(dataclasses.astuple(self))


class SplitItemDataclass:
    """..."""
    @classmethod
    def parse(cls, s):
        fields = dataclasses.fields(cls)
        items = s.split(getattr(cls, '_sep', '-'))
        if len(items) > len(fields):
            raise ValueError(f'Too many items for {cls.__qualname__}: {s}')
        return cls(**{x.name: x.type(y) for x, y in zip(fields, items)})

    def __str__(self):
        items = (str(x) for x in dataclasses.astuple(self) if x is not None)
        return getattr(self, '_sep', '-').join(items)


@dataclasses.dataclass(frozen=True)
class ImageMode(ConvertibleDataclass, SplitItemDataclass):
    """..."""
    modality: str
    model: str
    param: str
    _sep = '-'


@dataclasses.dataclass(order=True, frozen=True)
class ImageTarget(ConvertibleDataclass, SplitItemDataclass):
    """..."""
    case: int
    scan: str
    lesion: int
    _sep = '-'


@dataclasses.dataclass(order=True, frozen=True)
class TextureSpec(ConvertibleDataclass, SplitItemDataclass):
    """Texture feature specification."""
    method: str
    winsize: str
    feature: str
    _sep = '-'


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
        return self > self.__class__(*threshold_seq)


# @classmethod
# def parse(cls, s):
#     return cls(*(int(x) for x in s.split('+')))

# def __str__(self):
#     return '+'.join(str(x) for x in filter(None, self._astuple()))

# @property
# def label33(self):
#     return self.label(3, 3)

# @property
# def label34(self):
#     return self.label(3, 4)
