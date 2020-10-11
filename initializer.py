import nltk
import os
from shutil import copyfile

nltk.download('wordnet')
nltk.download('omw')

# Create entry for Slovak ('slk')
resource_path = ".\\resources\slk_wordnet"
slk_path = nltk.data.find("corpora") + "\omw\slk"

try:
    os.mkdir(slk_path)
except FileExistsError:
    pass

copy_files = os.listdir(resource_path)
for file in copy_files:
    copyfile(resource_path + "\\" + file, slk_path + "\\" + file)
