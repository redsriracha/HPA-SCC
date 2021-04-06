from zipfile import ZipFile as zip
from tqdm import tqdm

data = zip('train.zip')
for item in tqdm(data.namelist()):
    if item.endswith('.png'):
        data.extract(item, 'train')

print('DONE')