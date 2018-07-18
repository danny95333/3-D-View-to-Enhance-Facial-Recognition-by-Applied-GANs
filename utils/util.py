import shutil

def ParseImgName(imgName):
    imgName = imgName.split('.')[0].split('_')
    return int(imgName[0]),int(imgName[3]),int(imgName[4]) # identity is 0 and pose is 3
