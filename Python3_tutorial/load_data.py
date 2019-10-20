from sklearn import preprocessing
import pandas as pd


def load_data(path):
    names = ['height', 'weight', 'bust', 'waist', 'hip', 'sex', 'user_id_md5', 'goods_id_md5', 'size_id_1', 'size_id_2',
             'sku_id_md5',
             'new_category_3rd_id', 'cate3_gender_id', 'name', 'f2', 'f5', 'f6', 'f9', 'f11', 'f12', 'f13', 'f14',
             'f15', 'name_adjust', 'f2_adjust',
             'f5_adjust', 'f6_adjust', 'f9_adjust', 'f11_adjust', 'f12_adjust', 'f13_adjust', 'f14_adjust',
             'f15_adjust', 'f2_adjust_height',
             'f2_adjust_peri', 'label']

    size_data = pd.read_csv(path, error_bad_lines=False,
                            header=None, names=names)

    # 删除列
    size_data.drop(size_data.columns[6:25], axis=1, inplace=True)
    size_data.drop(size_data.columns[9:16], axis=1, inplace=True)
    size_data.drop(size_data.columns[2:5], axis=1, inplace=True)

    # 修改列名
    size_data.rename(columns={'f5_adjust': 'shoulder', 'f6_adjust': 'bust', 'f9_adjust': 'clothes_length'},
                     inplace=True)

    # 转换列数据格式，强制空值为NaN
    size_data[['height', 'weight', 'sex']] = size_data[['height', 'weight', 'sex']].astype(float)
    size_data[['shoulder', 'bust', 'clothes_length', 'label']] = size_data[
        ['shoulder', 'bust', 'clothes_length', 'label']].apply(
        pd.to_numeric, errors='coerce')

    # 删除带有NaN的行
    size_data.dropna(axis=0, how='any', inplace=True)
    size_data.reset_index(drop=True, inplace=True)
    size_data[['label']] = size_data[['label']].astype(int)
    # 对行做shuffle
    size_data = size_data.sample(frac=1)
    size_data.reset_index(drop=True, inplace=True)
    # print(size_data)
    size_data = size_data[['height', 'weight', 'shoulder', 'bust', 'clothes_length', 'label']].values
    return size_data


path = 'E:/Desktop/Data/vip/vip_products_size_data/size_fc_v2/000001_0.txt'
test_data = load_data(path)
test_x = test_data[:, 0:-1]
test_y = test_data[:, -1:]
print(test_x)
print(test_y)
print(preprocessing.scale(test_x))
print(test_data.shape)
