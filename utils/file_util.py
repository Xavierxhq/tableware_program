import pickle


def pickle_read(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        print('Pickle read error: not exits {}'.format(file_path))
        return None


def pickle_write(file_path, what_to_write):
    try:
        with open(file_path, 'wb+') as f:
            pickle.dump(what_to_write, f)
    except:
        print('Pickle write error: {}'.format(file_path))


def write_csv(path, datas):
    import pandas as pd

    dataframe = pd.DataFrame(datas)
    dataframe.to_csv(path, index=False, sep=',', encoding="gb2312")
    print(path.split('/')[-1], 'saved.')
