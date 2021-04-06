import os
from zipfile import ZipFile as zip

PYTHON = 'python'
KAGGLE_INSTALL = f'{PYTHON} -m pip install kaggle'
KAGGLE_COMP_NAME = f'hpa-single-cell-image-classification'
KAGGLE_API = f'kaggle competitions download -c {KAGGLE_COMP_NAME}'
KAGGLE_ZIP = f'{KAGGLE_COMP_NAME}.zip'

print(KAGGLE_INSTALL)
os.system(KAGGLE_INSTALL)
print(KAGGLE_API)
os.system(KAGGLE_API)

data = zip(KAGGLE_ZIP)

EXTRACT_ITEMS = [
    'tfrecords',
    '.csv',
]

for item in data.namelist():
    if any(name in item for name in EXTRACT_ITEMS):
        data.extract(item, '.')