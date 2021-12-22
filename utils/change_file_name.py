import os
import re

def change_file_name(folder):
    for file in os.listdir(folder):
        os.rename(folder + '/' + file, folder + re.findall('.*PetImages(.*)', folder)[0] + file)

if __name__ == '__main__':
    print(os.listdir('./PetImages/Dog'))
    change_file_name('./PetImages/Dog')
    print(os.listdir('./PetImages/Dog'))
