def iou(gt_bbox, pred_bbox):
    """
    Cálcula el valor del índice de Jaccard, o IoU, entre dos bounding boxes.
    Parameters
    ----------
    gt_bbox: list
        Lista de 4 elementos que representan el bounding box real del objeto.
    pred_bbox: list
        Lista de 4 elementos que representan el bounding box predicho del objeto.

    Returns
    -------
    iou: float
        Valor del índice de Jaccard, o IoU.
    intersection: float
        Valor de la intersección.
    union: float
        Valor de la unión.
    """

    x1, y1, w1, h1 = gt_bbox
    x2, y2, w2, h2 = pred_bbox

    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)

    if w_intersection <= 0 or h_intersection <= 0:
        return 0, 0, 0

    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union, intersection, union


def f1score(gt_bbox, pred_bbox, smooth=1e-6):
    """
    Cálcula el valor del índice de Sorensen-Dice, o F1 score, entre dos bounding boxes.
    Parameters
    ----------
    gt_bbox: list
        Lista de 4 elementos que representan el bounding box real del objeto.
    pred_bbox: list
        Lista de 4 elementos que representan el bounding box predicho del objeto.
    smooth: float
        Valor para ajustar el cálculo del índice y evitar divisiones entre 0.

    Returns
    -------
    f1score: float
        Valor del índice de Sorensen-Dice, o F1 score.
    """

    iou_v = iou(gt_bbox, pred_bbox)
    intersection = iou_v[1]
    union = gt_bbox[2] * gt_bbox[3] + pred_bbox[2] * pred_bbox[3]
    dice = 2. * intersection / (union + smooth)
    return dice


def yolobbox2bbox(data, dw=1292, dh=120):
    _, x, y, w, h = map(float, data.split(' '))
    left = int((x - w / 2) * dw)
    right = int((x + w / 2) * dw)
    top = int((y - h / 2) * dh)
    bottom = int((y + h / 2) * dh)

    if left < 0:
        left = 0
    if right > dw - 1:
        right = dw - 1
    if top < 0:
        top = 0
    if bottom > dh - 1:
        bottom = dh - 1
    return [left, top, abs(right - left), abs(bottom - top)]

