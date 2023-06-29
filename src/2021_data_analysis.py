import pandas as pd
from matplotlib import pyplot, font_manager, rc
from pylab import *
import numpy

font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
data = pd.read_excel("데이터.xlsx")
print("Read all datas")


def divide_attributes():
    with open("attributes.txt",'+w',encoding="UTF-8") as att:
        with open("more_than_10.txt", '+w',encoding="UTF-8") as ten:
            for key in data.keys():
                if len(set(list(data[key]))) <= 10:
                    print("att => {}".format(key))
                    att.write("{} : {}\n".format(key, set(list(data[key]))))
                else:
                    print("too long => {}".format(key))
                    ten.write("{} : {}\n".format(key, list(set(list(data[key])))[:15]))
                            
            pass
        
def num_non_numeric(data : pd.DataFrame, key : str):
    key_elem = list(set(list(data[key])))
    temp = [0 for i in range(len(key_elem))]
    
    for elem in data[key]:
        for i, k in enumerate(key_elem):
            if elem == k:
                temp[i] += 1
    
    return temp, key_elem

def plot_non(axis : list, data : list, key : str):
    if len(data) != len(axis):
        IndexError("Data length and axis length is not same")
    plt.title(key)
    plt.bar(axis, data)
    
    for i, v in enumerate(axis):
        plt.text(v, data[i], data[i],                 
                fontsize = 9, 
                color='black',
                horizontalalignment='center',
                verticalalignment='bottom') 
    plt.savefig("{}.png".format(key))
    plt.show()
    plt.close()
    
    pass

def plot_num(axis : list, data : list, key : str):
    if len(data) != len(axis):
        IndexError("Data length and axis length is not same")
    plt.title(key)
    plt.plot(axis, data)
    plt.savefig("{}.png".format(key))
    plt.show()
    plt.close()
                    
# divide_attributes()
def statistics():
    non_numeric_atts = ['설립근거법', '설립유형', '공익사업유형', '설립주체','기부금(단체) 유형', '복식부기', '세무확인', '외부회계감사 여부']
    # numeric_atts = ['순자산비율','물질지분비율','기타자산비율','공익사업비율','기타사업비율','개인기부비율','영리기부비율','공익법인지원비율','기타기부비율','국내비율','국외비율','잉여기부비용']
    # for key in non_numeric_atts:
    #     t, k = num_non_numeric(data=data, key = key)
    #     print(key)
    #     print(k)
    #     print(t)
    #     plot_non(k,t, key=key)
        
    # for key in numeric_atts:
    #     t = sort(list(data[key]))
    #     plot_num(list(range(0,11463)), t, key=key)
    
    numeric_atts = ["자원봉사자 연인원 수"]
    for key in numeric_atts:
        t =numpy.log(sort(list(data[key])) + 1e-6)
        plot_num(list(range(0,11463)), t, key=key)
        # t = numpy.log(sort(list(data["기부금사용비율"])) + 1e-6)
        # plot_num(list(range(0,11463)), t,key="기부금사용비율")
        
        
        
def to_numpy():
    # for each x element, elem = []
    x = []
    for idx in range(11463):
        elem = data.loc[idx]
        # 설립근거법 = 사회복지법 : 
        
        # 교육 : 0, 기타 : 1, 의료 : 2, 학술?장학 : 3, 사회복지 : 4, 예술?문화 : 5
        
        
    
#     pass
# statistics()
# elem = data.loc[11462,]
# print(elem["공익사업유형"])
# print(len(set(list(data['설립유형']))))

