# Runnable script to download and install  Wordnet, OMW Wordnets and Slovak WordNet to device

import nltk
import os
from shutil import copyfile

# Download WordNet library
nltk.download('wordnet')
# Download OMW WordNets
nltk.download('omw')

# Create entry for Slovak ('slk')
resource_path = ".\\resources\slk_wordnet"
slk_path = nltk.data.find("corpora") + "\omw\slk"

# Create folder for Slovak WordNet
try:
    os.mkdir(slk_path)
except FileExistsError:
    pass

# Copy Slovak WordNet files (from this repository) to where they need to be
copy_files = os.listdir(resource_path)
for file in copy_files:
    copyfile(resource_path + "\\" + file, slk_path + "\\" + file)
