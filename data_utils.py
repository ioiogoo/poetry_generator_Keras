# *-* coding:utf-8 *-*
'''
@author: ioiogoo
@date: 2018/1/31 19:30
'''

puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']


def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 每行的末尾加上"]"符号代表一首诗结束
            for char in puncs:
                line = line.replace(char, "")
            files_content += line.strip() + "]"

    words = sorted(list(files_content))
    words.remove(']')
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    del counted_words[']']
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    # word到id的映射
    word2num = dict((c, i + 1) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, 0)
    return word2numF, num2word, words, files_content
