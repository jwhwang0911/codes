import pandas as pd
from matplotlib import pyplot, font_manager, rc
from pylab import *
from features import *
import numpy

data = pd.read_excel("../데이터.xlsx")
save_path = "../processed_datas.npy"
print("Read all datas")


def discrete_transform(index, data):
    match index:
        case "공익사업유형":
            match data:
                case "의료":
                    return 1.0
                case "사회복지":
                    return 2.0
                case "예술?문화":
                    return 3.0
                case "학술?장학":
                    return 4.0
                case "교육":
                    return 5.0
                case "기타":
                    return 6.0
        case "복식부기":
            match data:
                case "Y":
                    return 1.0
                case "N":
                    return 0.0
        case "설립유형":
            match data:
                case "공공기관":
                    return 0.1
                case "재단법인":
                    return 0.2
                case "기타단체":
                    return 0.3
                case "법인으로보는단체":
                    return 0.4
                case "사단법인":
                    return 0.5
        case "외부회계감사 여부":
            match data:
                case "Y":
                    return 1.0
                case "N":
                    return 0.0
    pass

def for_log(data):
    return numpy.log(data + 1e-6)


def construct_numpy():
    print("Construct numpy")
    np_dict = []

    for idx in range(11463):
        temp = []
        print("\r{}th data fin.".format(idx), end = "")
        for key in important_discrete_att:
            elem = data.loc[idx]
            # print(discrete_transform(key,elem[key]), end="\t")
            try:
                temp.append(discrete_transform(key, elem[key]))
            except:
                temp.append(-100.0)
        for key in important_numeric_att:
            elem = data.loc[idx]
            try:
                temp.append(elem[key])
            except:
                temp.append(-100.0)
            # print(elem[key], end="\t")
        for key in log_transform:
            elem = data.loc[idx]
            # print(for_log(elem[key]), end="\t")
            try:
                temp.append(for_log(elem[key]))
            except:
                temp.append(-100.0)
        print(len(temp))
        np_dict.append(temp)
    
    return np.array(np_dict)

def load_data():
    load_dict = np.load(save_path)
    return load_dict

# numpy_dict = construct_numpy()
# print(numpy_dict.shape)
# np.save(save_path, numpy_dict)
# print("Successfully saved")

print(load_data().shape)
