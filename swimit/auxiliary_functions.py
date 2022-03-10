import re
from constants.split_constants import SplitValues

def countours_union(a,b):
    """
    Obtiene el rectángulo que contiene a otros dos rectángulos dados.
    Parameters
    ----------
    a : list
        Primer rectángulo, compuesto por coordenadas X, Y, ancho y alto.
    b: list
        Segundo rectángulo, compuesto por coordenadas X, Y, ancho y alto.
    Returns
    -------
    (x,y,w,h) : list
        Rectángulo que contiene a los pasados por parámetro, compuesto por coordenadas X, Y, ancho y alto.
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def calculate_splits(videoname):
    """
    Obtiene el número de splits de la prueba.
    Parameters
    ----------
    videoname : str
        Nombre del fichero con la grabación de la prueba.
    Returns
    -------
    splits : int
        Número de splits de la prueba.
    """
    splits = 0
    match = re.search(SplitValues.REGEX_PATTERN,videoname)
    if match:
        total_distance = match.group(1)           
        splits = int(total_distance) // SplitValues.DISTANCE      
    return splits