from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionValues:
    """
    Objeto que almacenará las posibles resoluciones de los vídeos.

    Attributes
    ----------
    NORMAL_FRAME_RATE: int
        Frame rate de los vídeos grabados en resolución normal.
    HALF_RESOLUTION_WIDTH: int
        Ancho en píxeles de los vídeos a media resolución.
    HALF_RESOLUTION_HEIGHT: int
        Alto en píxeles de los vídeos a media resolución.
    """

    NORMAL_FRAME_RATE = 42
    HALF_WIDTH = 1292
    HALF_HEIGHT = 816
