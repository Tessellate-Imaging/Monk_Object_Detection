import os
import sys
import numpy as np 

def update(cfg_file, num_classes):
    name = cfg_file.split(".")[0];

    f = open(cfg_file, 'r');
    wr = f.read();
    f.close();


    line1 = "classes=80";
    line2 = "filters=255";

    updated_line1 = "classes=" + str(num_classes);
    updated_line2 = "filters=" + str((5 + num_classes)*3);

    wr = wr.replace(line1, updated_line1);
    wr = wr.replace(line2, updated_line2);

    f = open(cfg_file, 'w');
    f.write(wr)
    f.close();