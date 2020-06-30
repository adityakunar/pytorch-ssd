import untangle
import os

all_classes = [
    "train",
    "sofa",
    "chair",
    "car",
    "bicycle",
    "motorbike",
    "person",
    "pottedplant",
    "boat",
    "dog",
    "cat",
    "bottle",
    "aeroplane",
    "cow",
    "sheep",
    "tvmonitor",
    "bird",
    "diningtable",
    "bus",
    "horse"
]

def get_size_perc(file_name: str) -> (list, list):

    # file_name should be given as "000025"

    root = untangle.parse("./VOC0712/test/VOC2007/Annotations/" + file_name + ".xml")

    total_height = int(root.annotation.size.height.cdata)
    total_width = int(root.annotation.size.width.cdata)

    total_area = total_width * total_height

    obj_sizes = []
    classes = []

    for obj in root.annotation.object:
        width = abs(int(obj.bndbox.xmax.cdata) - int(obj.bndbox.xmin.cdata))
        height = abs(int(obj.bndbox.ymax.cdata) - int(obj.bndbox.ymin.cdata))
        area = width * height
        obj_sizes.append( float(width * height) / total_area)
        classes.append(obj.name.cdata)

    return obj_sizes, classes

def only_this_size(l: list, lower_bound: float, higher_bound: float) -> bool:
    for element in l:
        if element < lower_bound or element > higher_bound:
            return False
    return True

def replace_file(path: str, list_of_correct_images: list):
    list_of_correct_images = [string.replace(".xml", "\n") for string in list_of_correct_images]

    os.remove(path)

    with open(path, "w") as file:
        file.writelines(list_of_correct_images)

    return

def add_to_list(new_elements: list, current_list: list) -> list:
    for element in new_elements:
        if element not in current_list:
            current_list.append(element)

    return current_list

def missing(classes: list) -> list:
    miss = []
    for c in all_classes:
        if c not in classes:
            miss.append(c)
    return miss


images_00_05 = []
images_05_10 = []
images_10_20 = []
images_20_40 = []
images_40_60 = []
images_60_80 = []
images_80_99 = []

images_00_05_classes = []
images_05_10_classes = []
images_10_20_classes = []
images_20_40_classes = []
images_40_60_classes = []
images_60_80_classes = []
images_80_99_classes = []

i = 0
for filename in os.listdir("./VOC0712/test/VOC2007/Annotations/"):

    if i % 1000 == 0:
        print(filename)

    no_extension = filename.replace(".xml", "")
    sizes, classes = get_size_perc(no_extension)

    if only_this_size(sizes, 0.0, 0.05):
        images_00_05.append(filename)
        images_00_05_classes = add_to_list(classes, images_00_05_classes)
    if only_this_size(sizes, 0.05, 0.10):
        images_05_10.append(filename)
        images_05_10_classes = add_to_list(classes, images_05_10_classes)
    if only_this_size(sizes, 0.10, 0.20):
        images_10_20.append(filename)
        images_10_20_classes = add_to_list(classes, images_10_20_classes)
    if only_this_size(sizes, 0.20, 0.40):
        images_20_40.append(filename)
        images_20_40_classes = add_to_list(classes, images_20_40_classes)
    if only_this_size(sizes, 0.40, 0.60):
        images_40_60.append(filename)
        images_40_60_classes = add_to_list(classes, images_40_60_classes)
    if only_this_size(sizes, 0.60, 0.80):
        images_60_80.append(filename)
        images_60_80_classes = add_to_list(classes, images_60_80_classes)
    if only_this_size(sizes, 0.80, 1.00):
        images_80_99.append(filename)
        images_80_99_classes = add_to_list(classes, images_80_99_classes)
    i += 1

print("DONE WITH IMAGE EVALUATION")
print("=======================================")
print("There are", len(images_00_05), "images with size in [0.00, 0.05]")
print("CLASSES MISSING:", missing(images_00_05_classes))
print("There are", len(images_05_10), "images with size in [0.05, 0.10]")
print("CLASSES MISSING:", missing(images_05_10_classes))
print("There are", len(images_10_20), "images with size in [0.10, 0.20]")
print("CLASSES MISSING:", missing(images_10_20_classes))
print("There are", len(images_20_40), "images with size in [0.20, 0.40]")
print("CLASSES MISSING:", missing(images_20_40_classes))
print("There are", len(images_40_60), "images with size in [0.40, 0.60]")
print("CLASSES MISSING:", missing(images_40_60_classes))
print("There are", len(images_60_80), "images with size in [0.60, 0.80]")
print("CLASSES MISSING:", missing(images_60_80_classes))
print("There are", len(images_80_99), "images with size in [0.80, 1.00]")
print("CLASSES MISSING:", missing(images_80_99_classes))
print("=======================================")


counter = 0
for filename in os.listdir("./VOC0712/test/images_00_05/Annotations"):
    if filename not in images_00_05:
        os.remove("./VOC0712/test/images_00_05/Annotations/" + filename)
        os.remove("./VOC0712/test/images_00_05/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_00_05/present_classes.txt", "w") as file:
    for c in images_00_05_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_00_05/ImageSets/Main/test.txt", images_00_05)
print("Updated test.txt file")
print("Removed", counter, "images in [0.00, 0.05]")
print("not removed:", images_00_05[0:10])

counter = 0
for filename in os.listdir("./VOC0712/test/images_05_10/Annotations"):
    if filename not in images_05_10:
        os.remove("./VOC0712/test/images_05_10/Annotations/" + filename)
        os.remove("./VOC0712/test/images_05_10/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_05_10/present_classes.txt", "w") as file:
    for c in images_05_10_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_05_10/ImageSets/Main/test.txt", images_05_10)
print("Updated test.txt file")
print("Removed", counter, "images in [0.05, 0.10]")
print("not removed:", images_05_10[0:10])

counter = 0
for filename in os.listdir("./VOC0712/test/images_10_20/Annotations"):
    if filename not in images_10_20:
        os.remove("./VOC0712/test/images_10_20/Annotations/" + filename)
        os.remove("./VOC0712/test/images_10_20/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_10_20/present_classes.txt", "w") as file:
    for c in images_10_20_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_10_20/ImageSets/Main/test.txt", images_10_20)
print("Updated test.txt file")
print("Removed", counter, "images in [0.10, 0.20]")
print("not removed:", images_10_20[0:10])

counter = 0
for filename in os.listdir("./VOC0712/test/images_20_40/Annotations"):
    if filename not in images_20_40:
        os.remove("./VOC0712/test/images_20_40/Annotations/" + filename)
        os.remove("./VOC0712/test/images_20_40/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_20_40/present_classes.txt", "w") as file:
    for c in images_20_40_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_20_40/ImageSets/Main/test.txt", images_20_40)
print("Updated test.txt file")
print("Removed", counter, "images in [0.20, 0.40]")
print("not removed:", images_20_40[0:10])

counter = 0
for filename in os.listdir("./VOC0712/test/images_40_60/Annotations"):
    if filename not in images_40_60:
        os.remove("./VOC0712/test/images_40_60/Annotations/" + filename)
        os.remove("./VOC0712/test/images_40_60/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_40_60/present_classes.txt", "w") as file:
    for c in images_40_60_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_40_60/ImageSets/Main/test.txt", images_40_60)
print("Updated test.txt file")
print("Removed", counter, "images in [0.40, 0.60]")
print("not removed:", images_40_60[0:10])

counter = 0
for filename in os.listdir("./VOC0712/test/images_60_80/Annotations"):
    if filename not in images_60_80:
        os.remove("./VOC0712/test/images_60_80/Annotations/" + filename)
        os.remove("./VOC0712/test/images_60_80/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_60_80/present_classes.txt", "w") as file:
    for c in images_60_80_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_60_80/ImageSets/Main/test.txt", images_60_80)
print("Updated test.txt file")
print("Removed", counter, "images in [0.60, 0.80]")
print("not removed:", images_60_80[0:10])

counter = 0
for filename in os.listdir("./VOC0712/test/images_80_99/Annotations"):
    if filename not in images_80_99:
        os.remove("./VOC0712/test/images_80_99/Annotations/" + filename)
        os.remove("./VOC0712/test/images_80_99/JPEGImages/" + filename.replace(".xml", ".jpg"))
        counter += 1

with open("./VOC0712/test/images_80_99/present_classes.txt", "w") as file:
    for c in images_80_99_classes:
        file.write(c + "\n")
replace_file("./VOC0712/test/images_80_99/ImageSets/Main/test.txt", images_80_99)
print("Updated test.txt file")
print("Removed", counter, "images in [0.80, 0.99]")
print("not removed:", images_80_99[0:10])