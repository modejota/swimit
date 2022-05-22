import sys
from tkinter import Tk, filedialog, simpledialog


class UI:

    @staticmethod
    def askforvideofile():
        Tk().withdraw()
        try:
            vid = filedialog.askopenfilename(initialdir="../sample_videos", title="Seleccione fichero", filetypes=[
                ("Video files", ".avi .mp4 .flv"),
                ("AVI file", ".avi"),
                ("MP4 file", ".mp4"),
                ("FLV file", ".flv"),
            ])
        except(OSError, FileNotFoundError):
            print(f'No se ha podido abrir el fichero seleccionado.')
            sys.exit(100)
        except Exception as error:
            print(f'Ha ocurrido un error: <{error}>')
            sys.exit(101)
        if len(vid) == 0 or vid is None:
            print(f'No se ha seleccionado ningún archivo.')
            sys.exit(102)
        return vid

    @staticmethod
    def askforlanenumber():
        Tk().withdraw()
        lane = simpledialog.askinteger(title="Calle", prompt="Calle (1-8):", minvalue=1, maxvalue=8)
        if lane is None:
            print(f'No se ha seleccionado ninguna calle a analizar.')
            sys.exit(103)
        return lane

    @staticmethod
    def askforshowprocessing():
        Tk().withdraw()
        show = ""
        while show != "S" and show != "N" and show is not None:
            show = simpledialog.askstring(title="Mostrar", prompt="Mostrar procesamiento (S/N):")
            if show is None:
                print(f'No se ha seleccionado ninguna opción.')
                sys.exit(102)
        return True if show == "S" else False

    @staticmethod
    def askforsavegraphs():
        Tk().withdraw()
        save = ""
        while save != 'S' and save != 'N' and save is not None:
            save = simpledialog.askstring(title="Guardar", prompt="Guardar gráficas (S/N):")
            if save is None:
                print(f'No se ha seleccionado ninguna opción.')
                sys.exit(102)
        return True if save == "S" else False
