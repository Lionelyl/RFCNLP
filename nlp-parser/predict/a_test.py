#
# import my_data_utils
# files = ["../rfcs-original/for_predict/TCP_original_for_annotation.txt"]
# word2id = {}; tag2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}
# X_pred_data = my_data_utils.get_data(files, word2id, id2word, id2cap)
# print(X_pred_data)

name = 'BGPv4'
file = './rfc/{}_annotation.txt'.format(name)

text = ""
flag = False
# with open(file, 'r') as fp:
#     for line in fp:
#         # if "State(s):" not in line and not flag:
#         #     continue
#         if "State(s):" in line:
#             flag = True
#             text += '\n' + line.strip() + '\n'
#         elif line == '\n':
#             continue
#         else:
#             text += line.strip() + '\n'
#         if "9.4" in line:
#             text += line.strip() + '\n'
#             break

text = ""
flag = False
states = [	"Idle",
            "Connect",
          	"Active"	,
          	"OpenSent"	,
          	"OpenConfirm"	,
          	"Established"	,]
new_block = False
new_inline = False
first_line = True
with open(file, 'r') as fp:
    for line in fp:
        if first_line:
            pre_line = line
            first_line = False
            continue
        if line.startswith(('Rekhter','', 'RFC')):
            continue
        if pre_line != '\n' and pre_line[3] != ' ' :
            new_block = True
            text += '======\n' + pre_line
            pre_line = line
        else:
            text += pre_line
            pre_line = line
        # elif new_block:
        #     if line[3] != ' ' :
        #         new_block = False
        #     new_block = True
        #     text += pre_line
        #     pre_line = line

        # if "State(s):" not in line and not flag:
        #     continu
print(text)
new_file = './rfc/{}_annotation_new.txt'.format(name)
with open(new_file,'w') as fw:
    fw.write(text
             )