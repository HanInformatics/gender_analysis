#2018/09 created
#2019.03 updated

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
system_name = platform.system()
print(system_name)

def set_Han_font():
    if system_name == 'Darwin': #Mac
        #download https://github.com/ubermenschjo/dotfiles/blob/master/.fonts/HANBatang.ttf to '/Library/Fonts'
        #rm /Users/myid/.matplotlib/*
        '''import matplotlib
        print(matplotlib.get_cachedir())'''
        #refer to https://github.com/matplotlib/matplotlib/issues/8427/
        font_name = fm.FontProperties(fname='/Library/Fonts/HanBatang.ttf').get_name()
    elif system_name == 'Linux':
        #sudo apt-get install fonts-unfonts-core fonts-unfonts-extra
        #rm -rf ~/.cache/matplotlib/*
        font_name = fm.FontProperties(fname='/usr/share/fonts/truetype/unfonts-core/UnBatang.ttf').get_name()
    else: #Windows
        #download hangul fonts from https://www.hancom.com/cs_center/csDownload.do
        #decompress and copy *.ttf into font directory below.
        font_name = fm.FontProperties(fname='C:/Windows/Fonts/HANBatang.ttf').get_name()
    plt.rc('font', family=font_name)
    return plt

def draw_bar_chart(words_freqs):
    words = [w for w, _ in words_freqs]
    freqs = [f for _, f in words_freqs]
    xs=[i+0.5 for i, _ in enumerate(words)]
    plt.bar(xs, freqs)
    plt.xticks(xs, words)
    plt.ylabel('# of words')
    plt.xlabel('top %s occurred words' %len(words_freqs))
    plt.show()
    return

if __name__ == "__main__":
    try:
        set_Han_font()
        draw_bar_chart([('한글',3), ('한국학',5), ('Korea',1)])
    except Exception as e:
        print(e)
