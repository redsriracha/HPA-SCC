import csv
import numpy as np

FILE = "train.csv"
INDEX = 1
DELIMITER = '|'
NUM_CLASS = 19

csvfile = open(FILE)
class_count = np.zeros(NUM_CLASS)

# Skip header row
next(csvfile)

for row in csv.reader(csvfile):
    items = row[INDEX].split(DELIMITER)
    for i in items:
        class_count[int(i)] += 1

print(class_count)