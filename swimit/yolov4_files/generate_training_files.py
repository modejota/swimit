import os
import glob

# Directorio actual. Mover a yolov4/darknet/
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = 'data/obj'

percentage_test = 15

file_train = open('data/train.txt', 'w')
file_valid = open('data/valid.txt', 'w')

# Rellenamos el fichero de validacion y entrenamiento con el número de imágenes que corresponda.
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_valid.write("data/obj" + "/" + title + '.jpg' + "\n")
    else:
        file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
        counter = counter + 1
