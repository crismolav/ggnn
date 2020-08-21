import json
from pdb import set_trace
import os
import sys
import random
# from bert_embedding import BertEmbedding
import numpy as np
from numpy import save, load
POS_list = [
    'JJ', 'NNS', 'IN', 'DT', 'NNP',
    'NNPS', 'CC', 'VBD', 'VBZ', 'NN', 'WDT', 'MD',
    'VB', 'WRB', 'POS', 'WP$' ,'-LRB-', '-RRB-',
    '.', ',', ':', 'VBG', 'TO', 'VBP', 'RB', "``", "''",
    "RBR", "$", "CD", "VBN", "JJR" ,"RP", "PRP", 'PRP$',
    'PDT', 'RBS', 'WP', 'JJS', 'EX', 'UH', 'FW', '#', 'LS']

sample_dep_list = ['ROOT', 'det', 'nsubj', 'aux']
# bert_embedding = BertEmbedding()

def process_sentence(sentence_list, dep_list, problem='root', sentence_list_out=None,
                     dep_list_out=None, sentence_id=None, pos_list=None, word_dict=None,
                     all_words_dict=None):
    graph = []
    target_list = []
    sentence_pos_list = []
    sentence_pos_list_out = []
    sentence_dict = {}
    words_index_list = []
    all_words_index_list = []
    target_id = None
    dependencies_set = set()
    selected_id = get_random_node(sentence_list)
    last_node = sentence_list[-1]

    for i, line in enumerate(sentence_list):
        line_as_list = line.split('\t')
        id = line_as_list[0]
        word = line_as_list[1].lower()
        pos = line_as_list[3]
        father = line_as_list[6]
        dep = line_as_list[7].strip()
        ref_dep_list = sample_dep_list if problem == 'id_sample' else dep_list
        #edges start from 1 not from 0
        edge_type = ref_dep_list.index(dep) + 1
        new_edge = [int(father), edge_type, int(id)]
        graph.append(new_edge)
        sentence_pos_list.append(pos)
        word_index = word_dict[word] if word in word_dict else word_dict['NA']
        words_index_list.append(word_index)

        all_words_index = all_words_dict[word]
        all_words_index_list.append(all_words_index)

        if problem == 'root' and int(father) == 0:
            target_id = int(id)
            target_list.append([target_id])
        elif problem == 'head' and int(id) == selected_id:
            target_id = int(father)
            target_list.append(get_node_as_vector(id=target_id, last_node=last_node))
        elif problem in ['identity','id_sample', 'btb_id']:
            target_unit = [int(father), int(edge_type)]
            target_list.append(target_unit)
        elif problem in ['btb', 'btb_sample']:
            line_out = sentence_list_out[i]
            line_as_list_out = line_out.split('\t')
            pos_out = line_as_list_out[3]
            father_out = line_as_list_out[6]
            dep_out = line_as_list_out[7].strip()
            edge_type_out = dep_list_out.index(dep_out) + 1
            target_unit = [int(father_out), int(edge_type_out)]
            target_list.append(target_unit)
            sentence_pos_list_out.append(pos_out)

        dependencies_set.add(dep)

    sentence_dict["targets"] = target_list
    sentence_dict["graph"]   = graph
    sentence_dict["node_features"] = get_node_features(
        problem=problem, selected_id=selected_id,
        target_id=target_id, sentence_list=sentence_list,
        sentence_pos_list=sentence_pos_list,
        pos_list=pos_list)
    sentence_dict["node_features_target"] = get_node_features(
        problem=problem, selected_id=selected_id,
        target_id=target_id, sentence_list=sentence_list,
        sentence_pos_list=sentence_pos_list_out,
        pos_list=pos_list)
    sentence_dict["raw_sentence"] = ' '.join([x.split('\t')[1] for x in sentence_list])
    sentence_dict["id"] = str(sentence_id).zfill(5)
    #we add the zero index to the node zero
    sentence_dict["words_index"] = [0] + words_index_list
    sentence_dict["bert_words_index"] = [0] + all_words_index_list

    return sentence_dict

def get_node_pos_list(pos_list, sentence_pos_list):
    pos_list = [0] + [pos_list.index(x) for x in sentence_pos_list]

    return  pos_list
def get_random_node(sentence_list):
    random.seed(0)
    random_line = random.choice(sentence_list)
    line_as_list = random_line.split('\t')
    id = line_as_list[0]
    return int(id)


def get_node_as_vector(id, last_node):
    last_id = int(last_node.split('\t')[0])
    node_as_vector = [0] * (last_id + 1)
    node_as_vector[int(id)] = 1
    return node_as_vector

def get_embeddings(value, type):
    try:
        value_index = None
        if type == 'pos':
            value_index = POS_list.index(value)


        node_feature = [0] * len(POS_list)
        node_feature[value_index] = 1
        return node_feature

    except:
        print("%s %s not in list"%(type.upper(), value))
def get_node_feature(problem, pos, id):
    if problem == 'root':
        try:
            return [int(id)]
            # pos_index = POS_list.index(pos)
            # node_feature = [0]*len(POS_list)
            # node_feature[pos_index] = 1
            # return node_feature
        except:
            print("POS %s not in list"%pos)

def get_node_features(problem, selected_id, target_id, sentence_list,
                      sentence_pos_list=None, pos_list=None):
    last_node = sentence_list[-1]
    last_id = int(last_node.split('\t')[0])
    node_features = []

    if problem == 'root':
        node_features = [[x] for x in range(0, last_id+1)]
    elif problem in ['head']:
        node_features = [[0, 0]]*(last_id+1)
        node_features[selected_id] = [1, 0]
        node_features[target_id] = [0, 1]
    elif problem in ['identity', 'id_sample']:
        node_features = [[0] for _ in range(0, last_id+1)]
    elif problem in ['btb', 'btb_id', 'btb_sample']:
        #we add node zero
        node_features = [0] + [pos_list.index(x) for x in sentence_pos_list]
    return node_features

def get_new_file_path(problem, file_name):

    new_file_folder = '/Users/cristianmorales/Documents/Classes/Thesis/gated-graph-neural-network-samples'
    new_file_path = None

    if problem == 'root':
        new_file_name = file_name.replace(".conll", ".json")
        new_file_path = '%s/%s' % (new_file_folder, new_file_name)

    elif problem == 'head':
        new_file_name = file_name.replace(".conll", "_head.json")
        new_file_path = '%s/%s' % (new_file_folder, new_file_name)


    elif problem in ['identity', 'id_sample']:
        new_file_name = file_name.replace(".conll", "_id.json")
        new_file_path = '%s/%s' % (new_file_folder, new_file_name)

    elif problem in ['btb', 'btb_id']:
        new_file_name = file_name.replace(".conll", "_btb.json")
        new_file_path = '%s/%s' % (new_file_folder, new_file_name)
    elif problem in ['btb_sample']:
        new_file_name = file_name.replace(".conll", "_btb_sample.json")
        new_file_path = '%s/%s' % (new_file_folder, new_file_name)

    return new_file_path

def get_file_path(file_name):
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    # file_name = 'en-wsj-std-dev-stanford-3.3.0-tagged.conll'
    file_path = '%s/%s' % (path, file_name)
    return file_path

def get_dep_list(bank_type):
    if bank_type == 'nivre':
        file_names = ['en-wsj-ym-nivre-dev.conll', 'en-wsj-ym-nivre-test.conll',
                      'en-wsj-ym-nivre-train.conll']
    else:
        file_names = ['en-wsj-std-dev-stanford-3.3.0-tagged.conll',
                      'en-wsj-std-test-stanford-3.3.0-tagged.conll',
                      'en-wsj-std-train-stanford-3.3.0.conll']
    dep_set = set()
    for file_name in file_names:
        file_path = get_file_path(file_name=file_name)
        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()
            for i, line in enumerate(lines):
                if line.strip() == '':
                    continue
                line_as_list = line.split('\t')
                dep = line_as_list[7].strip()
                dep_set.add(dep)

    dep_list = list(dep_set)
    dep_list.sort()
    return dep_list

def get_dep_and_pos_list(bank_type, sample_size=None):
    if bank_type == 'nivre':
        file_names = ['en-wsj-ym-nivre-dev.conll', 'en-wsj-ym-nivre-test.conll',
                      'en-wsj-ym-nivre-train.conll']
    else:
        file_names = ['en-wsj-std-dev-stanford-3.3.0-tagged.conll',
                      'en-wsj-std-test-stanford-3.3.0-tagged.conll',
                      'en-wsj-std-train-stanford-3.3.0.conll']
    dep_set = set()
    pos_set = set()
    word_set = set()
    all_words_set = set()
    max_nodes = 0

    for file_name in file_names:
        count = 0
        file_path = get_file_path(file_name=file_name)
        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()
            last_line = None
            for i, line in enumerate(lines):
                if line.strip() == '':
                    count += 1
                    nodes_number = int(last_line[0]) + 1
                    if nodes_number > max_nodes:
                        max_nodes = nodes_number
                    if sample_size is not None and count == sample_size:
                        break
                    continue
                line_as_list = line.split('\t')
                word = line_as_list[1].lower()
                pos = line_as_list[3]
                dep = line_as_list[7].strip()
                if file_name in ['en-wsj-std-train-stanford-3.3.0.conll', 'en-wsj-ym-nivre-train.conll']:
                    word_set.add(word)
                all_words_set.add(word)
                pos_set.add(pos)
                dep_set.add(dep)
                last_line = line_as_list


    dep_list = list(dep_set)
    pos_list = list(pos_set)
    word_list = list(word_set)
    all_words_set = list(all_words_set)
    dep_list.sort()
    pos_list.sort()
    word_list.sort()
    all_words_set.sort()
    pos_list = ['zero'] + pos_list
    word_list = ['zero_'] + word_list + ['NA']
    all_words_set = ['zero_'] + all_words_set
    vocab_size = len(word_list)

    return dep_list, pos_list, word_list, vocab_size, max_nodes, all_words_set

def get_input_output_file(file_type, input_bank):
    train_files = ('en-wsj-std-train-stanford-3.3.0.conll', 'en-wsj-ym-nivre-train.conll')
    dev_files = ('en-wsj-std-dev-stanford-3.3.0-tagged.conll', 'en-wsj-ym-nivre-dev.conll')
    test_files = ('en-wsj-std-test-stanford-3.3.0-tagged.conll', 'en-wsj-ym-nivre-test.conll')
    if file_type == 'train':
        if  input_bank == 'std':
            return train_files
        elif  input_bank == 'nivre':
            return train_files[1], train_files[0]

    elif file_type == 'dev':
        if  input_bank == 'std':
            return dev_files
        elif  input_bank == 'nivre':
            return dev_files[1], dev_files[0]
    else:
        if  input_bank == 'std':
            return test_files
        elif  input_bank == 'nivre':
            return test_files[1], test_files[0]

def get_bert_array(all_words_list):
    # bert_dict = {}
    bert_list = []
    max = -1
    count = 0
    # for token in all_words_list:
    #     bert_list.append(bert_embedding([token])[0][1][0])
    #     count+=1
    #     if max!=-1 and count == max:
    #         break

    bert_array = np.array(bert_list) # [n , 768]
    return bert_array

def main():
    problem = sys.argv[1]

    if problem in ['btb', 'btb_id', 'btb_sample']:
        file_type = sys.argv[2]
        assert file_type in ['train', 'dev', 'test']
        input_bank = sys.argv[3] if len(sys.argv) > 3 else None
        assert input_bank in ['std', 'nivre']
        file_name_in, file_name_out = get_input_output_file(file_type=file_type, input_bank=input_bank)

        sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
        file_path_in = get_file_path(file_name=file_name_in)
        file_path_out = get_file_path(file_name=file_name_out)
        new_file_path = get_new_file_path(problem=problem, file_name=file_name_in)

        count = 0
        output_bank = 'nivre' if input_bank == 'std' else 'std'
        dep_list_in, pos_list, word_list, _, _, all_words_list = get_dep_and_pos_list(bank_type=input_bank, sample_size=sample_size)
        dep_list_out, _, _, _, _, _= get_dep_and_pos_list(bank_type=output_bank, sample_size=sample_size)

        word_dict = {k: v for v, k in enumerate(word_list)}
        all_words_dict = {k: v for v, k in enumerate(all_words_list)}

        with open(file_path_in, 'r') as input_file_in:
            with open(file_path_out, 'r') as input_file_out:
                with open(new_file_path, 'w') as output_file:
                    lines_in = input_file_in.readlines()
                    lines_out = input_file_out.readlines()
                    new_sentence_list_in = []
                    new_sentence_list_out = []
                    output_file.write('[')
                    for i, line_in in enumerate(lines_in):
                        line_out = lines_out[i]
                        if line_in.strip() == '':
                            sentence_dict = process_sentence(
                                sentence_list=new_sentence_list_in, problem=problem,
                                dep_list=dep_list_in, sentence_list_out=new_sentence_list_out,
                                dep_list_out=dep_list_out, sentence_id=count, pos_list=pos_list,
                                word_dict=word_dict, all_words_dict=all_words_dict,
                                )

                            output_file.write(json.dumps(sentence_dict))
                            new_sentence_list_in = []
                            new_sentence_list_out = []
                            count += 1
                            if sample_size is not None and count == sample_size:
                                break
                            if i != len(lines_in) - 1:
                                output_file.write(', ')

                        else:
                            new_sentence_list_in.append(line_in)
                            new_sentence_list_out.append(line_out)

                    print("Number of samples in %s: %i" % (file_path_in, count))
                    output_file.write(']')

    elif problem in ['bert']:
        input_bank = sys.argv[2]
        assert input_bank in ['std', 'nivre']
        output_bank = 'nivre' if input_bank == 'std' else 'std'
        dep_list_in, pos_list, word_list, _, _, all_words_list = get_dep_and_pos_list(
            bank_type=input_bank, sample_size=None)
        file_path_bert = get_file_path(file_name='bert_array2.npy')
        # all_words_dict = {k: v for v, k in enumerate(all_words_list)}
        bert_array = get_bert_array(all_words_list)

        save(file_path_bert, bert_array)

    else:

        file_name = sys.argv[2]
        experiment = True if len(sys.argv) >=4 and sys.argv[3] == '1' else False
        # file_name = 'en-wsj-std-dev-stanford-3.3.0-tagged.conll'
        file_path = get_file_path(file_name=file_name)
        new_file_path = get_new_file_path(problem=problem, file_name=file_name)

        count = 0
        bank_type = 'nivre' if 'nivre' in file_name else 'std'
        dep_list, pos_list, word_list, _, _, _ = get_dep_and_pos_list(bank_type=bank_type)
        word_dict = {k: v for v, k in enumerate(word_list)}
        with open(file_path, 'r') as input_file:
            with open(new_file_path, 'w') as output_file:
                lines = input_file.readlines()
                new_sentence_list = []
                output_file.write('[')
                for i, line in enumerate(lines):
                    if line.strip() == '':
                        sentence_dict = process_sentence(
                            sentence_list=new_sentence_list, problem=problem, dep_list=dep_list,
                            sentence_id=i, pos_list=pos_list, word_dict=word_dict)

                        output_file.write(json.dumps(sentence_dict))
                        new_sentence_list = []
                        count += 1

                        if i != len(lines) - 1:
                            output_file.write(', ')
                        if experiment:
                            break
                    else:
                        new_sentence_list.append(line)

                print("Number of samples in %s: %i" % (file_name, count))
                output_file.write(']')


# def count_words_train():
#     file_name_train = 'en-wsj-std-train-stanford-3.3.0.conll'
#     file_path_train = get_file_path(file_name=file_name_train)
#     with open(file_path_train, 'r') as input_file:
#         lines = input_file.readlines()
#         for i, line in enumerate(lines):
#             if line.strip() == '':
#                 nodes_number = int(last_line[0]) + 1
#                 if nodes_number > max_nodes:
#                     max_nodes = nodes_number
#                 if sample_size is not None and count == sample_size:
#                     break
#                 continue
#             line_as_list = line.split('\t')
#             pos = line_as_list[3]
#             dep = line_as_list[7].strip()
#             pos_set.add(pos)
#             dep_set.add(dep)
#             last_line = line_as_list
        
    


if __name__ == "__main__":
    main()
    #(googlevenv) python to_graph.py identity en-wsj-std-dev-stanford-3.3.0-tagged.conll
    #(googlevenv) python to_graph.py btb dev nivre
