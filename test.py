from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

def nlp_test(test_filename,model):
    """
    Save labels to file
    """
    sen_list = []
    with open(test_filename) as inf:
        words = []
        for line in inf:
            line = line.strip()
            if(len(line) == 0):
                sen_list.append(words)
                words = []
            else:
                ls = line.split(' ')
                word = ls[0]
                words.append(word)
        #last line
        if words != []:
            sen_list.append(words)

    out_tag_list = []
    for words in sen_list:
        preds = model.predict(words)
        out_tag_list.append(preds)

    print(len(sen_list))
    print(len(sen_list[0]))
    print(len(out_tag_list))
    print(len(out_tag_list[0]))

    outf = open('test.eval', 'w+')
    for i in range(len(sen_list)):
        for j in range(len(sen_list[i])):
            if(len(sen_list[i]) != len(out_tag_list[i])):
                print("Oh! no\n")
            word, out_tag = sen_list[i][j], out_tag_list[i][j]
            line = word + ' '+ out_tag + '\n'
            outf.write(line)
        if(i != len(sen_list) - 1):
            outf.write('\n')
    outf.close()

def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    #for online test, save words and tags into file
    nlp_test('./data/test.eval', model)


if __name__ == "__main__":
    main()
