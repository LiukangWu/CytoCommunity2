import os
import shutil
import pandas as pd
import datetime
import random


# Hyperparameters
InputFolderName = "./Spleen_Input/"


## Below is for generation of pseudo-spatial maps by shuffling cell types in original REAL spatial maps.
ThisStep_OutputFolderName = "./Step0_Output/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)


def shuffle_lines(input_file_path, output_file_path):
    # Read all lines in the input file.
    with open(input_file_path, 'r') as file0:
        lines = file0.readlines()

    # Shuffle lines
    random.shuffle(lines)

    # Write all shuffled lines to the output file.
    with open(output_file_path, 'w') as file1:
        file1.writelines(lines)


print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("Generate a pseudo-spatial map for each sample...")

# Copy files of real samples/images from the input folder.
for filename in os.listdir(InputFolderName):
    if filename.endswith(".txt"):
        file_path = os.path.join(InputFolderName, filename)
        shutil.copy(file_path, ThisStep_OutputFolderName)

# Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
    Region_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["Image"],  # set our own names for the columns
)

for graph_index in range(0, len(region_name_list)):
    print(f"This is image-{graph_index}")

    region_name = region_name_list.Image[graph_index]

    # Construct CellTypeLabel for pseudo-samples/images.
    input_file_path = InputFolderName + region_name + '_CellTypeLabel.txt'
    output_file_path = ThisStep_OutputFolderName + region_name + '_pseudo_CellTypeLabel.txt'
    shuffle_lines(input_file_path, output_file_path)

    # Copy Coordinates for pseudo-samples/images.
    GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
    GraphCoord_path = ThisStep_OutputFolderName + region_name + "_pseudo_Coordinates.txt"
    shutil.copy(GraphCoord_filename, GraphCoord_path)

    # Construct GraphLabel for real and pseudo-samples/images.
    GraphLabel_file = ThisStep_OutputFolderName + '/' + region_name + "_GraphLabel.txt"
    GraphLabel_pseudo_file = ThisStep_OutputFolderName + '/' + region_name + "_pseudo_GraphLabel.txt"

    with open(GraphLabel_file, 'w') as file2:  #REAL as "1"
        file2.write('1')

    with open(GraphLabel_pseudo_file, 'w') as file3:  #PSEUDO as "0"
        file3.write('0')

# Refine ImageNameList.
Region_filename_refined = ThisStep_OutputFolderName + "ImageNameList.txt"

with open(Region_filename_refined, 'r') as file4:
    lines = file4.readlines()

new_lines = [line.strip() + '_pseudo\n' for line in lines]
all_lines = lines + new_lines

with open(Region_filename_refined, 'w') as file5:
    file5.writelines(all_lines)

print("Step0 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


