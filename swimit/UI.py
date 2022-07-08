import sys
from tkinter import Tk, filedialog, simpledialog
from swimit.constants.yolov4_paths_constants import YOLOv4Paths as YV4P


class UI:
    """ Clase con los métodos para interactuar con el usuario """

    @staticmethod
    def askforvideofile(initialdirectory="../samples_videos"):
        Tk().withdraw()
        try:
            vid = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero", filetypes=[
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
    def askforcfgfile(initialdirectory=YV4P.DEFAULT_YOLO_DIRECTORY):
        Tk().withdraw()
        try:
            cfg = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero",
                                             filetypes=[("Config files", ".cfg")])
        except(OSError, FileNotFoundError):
            print(f'No se ha podido abrir el fichero seleccionado.')
            sys.exit(100)
        except Exception as error:
            print(f'Ha ocurrido un error: <{error}>')
            sys.exit(101)
        if len(cfg) == 0 or cfg is None:
            print(f'No se ha seleccionado ningún archivo.')
            sys.exit(102)
        return cfg

    @staticmethod
    def askforweightsfile(initialdirectory=YV4P.DEFAULT_YOLO_DIRECTORY):
        Tk().withdraw()
        try:
            weights = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero",
                                                 filetypes=[("Weights files", ".weights")])
        except(OSError, FileNotFoundError):
            print(f'No se ha podido abrir el fichero seleccionado.')
            sys.exit(100)
        except Exception as error:
            print(f'Ha ocurrido un error: <{error}>')
            sys.exit(101)
        if len(weights) == 0 or weights is None:
            print(f'No se ha seleccionado ningún archivo.')
            sys.exit(102)
        return weights

    @staticmethod
    def askfornamesfile(initialdirectory=YV4P.DEFAULT_YOLO_DIRECTORY):
        Tk().withdraw()
        try:
            names = filedialog.askopenfilename(initialdir=initialdirectory, title="Seleccione fichero",
                                               filetypes=[("Names files", ".names")])
        except(OSError, FileNotFoundError):
            print(f'No se ha podido abrir el fichero seleccionado.')
            sys.exit(100)
        except Exception as error:
            print(f'Ha ocurrido un error: <{error}>')
            sys.exit(101)
        if len(names) == 0 or names is None:
            print(f'No se ha seleccionado ningún archivo.')
            sys.exit(102)
        return names

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

    @staticmethod
    def askforgpu():
        Tk().withdraw()
        use_gpu = ""
        while use_gpu != 'S' and use_gpu != 'N' and use_gpu is not None:
            use_gpu = simpledialog.askstring(title="Usar GPU", prompt="Usar GPU (S/N):")
            if use_gpu is None:
                print(f'No se ha seleccionado ninguna opción.')
                sys.exit(102)
        return True if use_gpu == "S" else False

    @staticmethod
    def askforoverride():
        Tk().withdraw()
        override = ""
        while override != 'S' and override != 'N' and override is not None:
            override = simpledialog.askstring(title="Sobrescribir", prompt="Sobrescribir datos (S/N):")
            if override is None:
                print(f'No se ha seleccionado ninguna opción.')
                sys.exit(102)
        return True if override == "S" else False
