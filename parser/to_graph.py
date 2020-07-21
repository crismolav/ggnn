import json
from pdb import set_trace
import os
import sys
import random

POS_list = [
    'JJ', 'NNS', 'IN', 'DT', 'NNP',
    'NNPS', 'CC', 'VBD', 'VBZ', 'NN', 'WDT', 'MD',
    'VB', 'WRB', 'POS', 'WP$' ,'-LRB-', '-RRB-',
    '.', ',', ':', 'VBG', 'TO', 'VBP', 'RB', "``", "''",
    "RBR", "$", "CD", "VBN", "JJR" ,"RP", "PRP", 'PRP$',
    'PDT', 'RBS', 'WP', 'JJS', 'EX', 'UH', 'FW', '#', 'LS']

sample_dep_list = ['ROOT', 'det', 'nsubj', 'aux']

def process_sentence(sentence_list, dep_list, problem='root',
                     sentence_list_out=None, dep_list_out=None, sentence_id=None):
    graph = []
    target_list = []
    sentence_dict = {}
    target_id = None
    dependencies_set = set()
    selected_id = get_random_node(sentence_list)
    last_node = sentence_list[-1]

    for i, line in enumerate(sentence_list):
        line_as_list = line.split('\t')
        id = line_as_list[0]
        label = line_as_list[1]
        pos = line_as_list[3]
        father = line_as_list[6]
        dep = line_as_list[7]
        ref_dep_list = sample_dep_list if problem == 'id_sample' else dep_list
        #edges start from 1 not from 0
        edge_type = ref_dep_list.index(dep) + 1
        new_edge = [int(father), edge_type, int(id)]
        graph.append(new_edge)

        if problem == 'root' and int(father) == 0:
            target_id = int(id)
            target_list.append([target_id])
        elif problem == 'head' and int(id) == selected_id:
            target_id = int(father)
            target_list.append(get_node_as_vector(id=target_id, last_node=last_node))
        elif problem in ['identity','id_sample']:
            target_unit = [int(father), int(edge_type)]
            target_list.append(target_unit)
        elif problem in ['btb']:
            line_out = sentence_list_out[i]
            line_as_list_out = line_out.split('\t')
            father_out = line_as_list_out[6]
            dep_out = line_as_list_out[7].strip()
            edge_type_out = dep_list_out.index(dep_out) + 1
            target_unit = [int(father_out), int(edge_type_out)]
            target_list.append(target_unit)


        dependencies_set.add(dep)

    # sentence_dict["targets"] = [[get_node_as_vector(
    #     id=target_id, last_node=last_node)]]
    sentence_dict["targets"] = target_list
    sentence_dict["graph"]   = graph
    sentence_dict["node_features"] = get_node_features(
        problem=problem, selected_id=selected_id,
        target_id=target_id, sentence_list=sentence_list)
    sentence_dict["raw_sentence"] = ' '.join([x.split('\t')[1] for x in sentence_list])
    sentence_dict["id"] = str(sentence_id).zfill(5)
    
    return sentence_dict

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

def get_node_features(problem, selected_id, target_id, sentence_list):
    last_node = sentence_list[-1]
    last_id = int(last_node.split('\t')[0])
    node_features = []

    if problem == 'root':
        node_features = [[x] for x in range(0, last_id+1)]
    elif problem in ['head']:
        node_features = [[0, 0]]*(last_id+1)
        node_features[selected_id] = [1, 0]
        node_features[target_id] = [0, 1]
    elif problem in ['identity', 'id_sample', 'btb']:
        node_features = [[0] for _ in range(0, last_id+1)]

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

    elif problem in ['btb']:
        new_file_name = file_name.replace(".conll", "_btb.json")
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
        file_names = ['en-wsj-ym-nivre-dev.conll', 'en-wsj-ym-nivre-test.conll']
    else:
        file_names = ['en-wsj-std-dev-stanford-3.3.0-tagged.conll',
                      'en-wsj-std-test-stanford-3.3.0-tagged.conll']
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
def main():
    problem = sys.argv[1]
    if problem == 'btb':
        file_name_in = sys.argv[2]
        file_name_out = sys.argv[3]
        file_path_in = get_file_path(file_name=file_name_in)
        file_path_out = get_file_path(file_name=file_name_out)

        new_file_path = get_new_file_path(problem=problem, file_name=file_name_in)

        count = 0

        dep_list_in = get_dep_list(bank_type='std')
        dep_list_out = get_dep_list(bank_type='nivre')

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
                                dep_list_out=dep_list_out, sentence_id=count)

                            output_file.write(json.dumps(sentence_dict))
                            new_sentence_list_in = []
                            new_sentence_list_out = []
                            count += 1

                            if i != len(lines_in) - 1:
                                output_file.write(', ')

                        else:
                            new_sentence_list_in.append(line_in)
                            new_sentence_list_out.append(line_out)

                    print("Number of samples in %s: %i" % (file_path_in, count))
                    output_file.write(']')
    else:

        file_name = sys.argv[2]
        experiment = True if len(sys.argv) >=4 and sys.argv[3] == '1' else False

        # file_name = 'en-wsj-std-dev-stanford-3.3.0-tagged.conll'
        file_path = get_file_path(file_name=file_name)

        new_file_path = get_new_file_path(problem=problem, file_name=file_name)

        count = 0
        bank_type = 'nivre' if 'nivre' in file_name else 'std'
        dep_list = get_dep_list(bank_type=bank_type)

        with open(file_path, 'r') as input_file:
            with open(new_file_path, 'w') as output_file:
                lines = input_file.readlines()
                new_sentence_list = []
                output_file.write('[')
                for i, line in enumerate(lines):
                    if line.strip() == '':
                        sentence_dict = process_sentence(
                            sentence_list=new_sentence_list, problem=problem, dep_list=dep_list,
                            sentence_id=i)

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

if __name__ == "__main__":
    main()
    #python to_graph.py identity en-wsj-std-dev-stanford-3.3.0-tagged.conll
    #python to_graph.py btb en-wsj-std-dev-stanford-3.3.0-tagged.conll en-wsj-ym-nivre-dev.conll
