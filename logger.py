"""Code for logging

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""
import time
import os

def getRed(skk):         return("\033[91m{}\033[00m" .format(skk))
def getGreen(skk):       return("\033[92m{}\033[00m" .format(skk))
def getYellow(skk):      return("\033[93m{}\033[00m" .format(skk))
def getLightPurple(skk): return("\033[94m{}\033[00m" .format(skk))
def getPurple(skk):      return("\033[95m{}\033[00m" .format(skk))
def getCyan(skk):        return("\033[96m{}\033[00m" .format(skk))
def getLightGray(skk):   return("\033[97m{}\033[00m" .format(skk))
def getBlack(skk):       return("\033[98m{}\033[00m" .format(skk))


class logger():
    def __init__(self, filepath,overrite=False):
        self.filepath = filepath
        self.colorfilepath = "{}_colored".format(filepath)
        print("Log file at : {}".format(self.filepath))
        if overrite or not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                f.write("Log created at {}\n".format(
                    time.strftime("%Y/%m/%d %H : %M : %S")))
            print("Log file created : {}".format(self.filepath))
        if overrite or not os.path.exists(self.colorfilepath):
            with open(self.colorfilepath, "w") as f:
                f.write("Log created at {}\n".format(
                    time.strftime("%Y/%m/%d %H:%M:%S")))
            print("Color Log file created : {}".format(self.colorfilepath))

    def log(self, content, include_header=False, show=True, end='\n'):
        with open(self.filepath, "a") as f:
            if include_header:
                f.write("[{}] ".format(time.strftime("%Y/%m/%d %H:%M:%S")))

            f.write("{}".format(content))
            f.write(end)

        with open(self.colorfilepath, "a") as f:
            if include_header:
                f.write("[{}] ".format(
                    getPurple(time.strftime("%Y/%m/%d %H:%M:%S"))))

            f.write("{}".format(content))
            f.write(end)
        if show:
            print(content, end=end)


if __name__ == "__main__":
    mylogger = logger("./logs/test.log")
    mylogger.log("aasdasd")
