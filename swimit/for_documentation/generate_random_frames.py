import os
import cv2
import numpy as np

from swimit.UI import UI
from swimit.constants.pool_constants import PoolValues as PV

# Utilizado de cara a la documentación del trabajo.

videoname = UI.askforvideofile("../../samples_videos")
video = cv2.VideoCapture(videoname)
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Seleccionamos 45 frames aleatorios para determinar manualmente "ground truth"
# Nos aseguramos parte del vídeo en el que haya nadador moviendose.
random_numbers = np.random.randint(frames*0.30, frames*0.80, 45)
random_numbers = np.sort(random_numbers)

borde_abajo_calle = PV.LANES_BOTTOM_PIXEL.get(str(UI.askforlanenumber()))
borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT

frames_leidos = 0
videoname_no_extension = videoname.split('.')[0]
file_path = os.path.abspath(os.path.dirname(__file__))
while video.isOpened():
    ok, original_frame = video.read()
    original_frame = original_frame[borde_abajo_calle:borde_arriba_calle, :, :]
    if ok:
        if frames_leidos in random_numbers:
            file_name = str(frames_leidos) + '.png'
            if not cv2.imwrite(os.path.join(file_path, file_name), original_frame):
                raise Exception('Error al guardar imagen')
        frames_leidos += 1
        if frames_leidos == frames:
            break

cv2.destroyAllWindows()
video.release()


