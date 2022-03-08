from dataclasses import dataclass

@dataclass(frozen=True)
class ResolutionValues:
    """
    Objeto que almacenará las posibles resoluciones de los vídeos.

    Attributes
    ----------
    HALF_RESOLUTION_WIDTH: int
        Ancho en píxeles de los vídeos a media resolución.
    HALF_RESOLUTION_HEIGHT: int
        Alto en píxeles de los vídeos a media resolución.
    FULL_RESOLUTION_WIDTH: int
        Ancho en píxeles de los vídeos a resolución máxima.
    FULL_RESOLUTION_HEIGHT: int
        Alto en píxeles de los vídeos a máxima resolución.
    """

    HALF_WIDTH = 1292  
    HALF_HEIGHT = 816

    FULL_WIDTH = 2584
    FULL_HEIGHT = 1632