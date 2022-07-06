from dataclasses import dataclass


@dataclass(frozen=True)
class SwimmerValues:
    """
    Objeto que almacenará diferentes parámetros propios de un nadador.

    Attributes
    ----------
    SWIMSUIT_MAX_WIDTH: int
        Ancho máximo del bañador.
    SWIMMER_HEIGHT_DIFFERENCE: int
        Diferencia en altura entre el tronco superior e inferior del nadador
    SWIMMER_MAX_WIDTH: int
        Ancho máximo del nadador. (Altura si estuviera de pie)
    SWIMMER_MIN_AREA: int
        Mínima área para considerar válido el contorno.
    SWIMMER_MAX_AREA: int
        Máxima área para considerar válido el contorno.
    SWIMMER_AVERAGE_LENGTH: int
        Longitud media del nadador.
    """

    SWIMSUIT_MAX_WIDTH = 22
    SWIMMER_HEIGHT_DIFFERENCE = 12
    SWIMMER_MAX_WIDTH = 115
    SWIMMER_MIN_AREA = 200
    SWIMMER_MAX_AREA = 2500
    SWIMMER_AVERAGE_LENGTH = 65
