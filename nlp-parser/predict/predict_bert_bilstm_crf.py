import os
import my_data_utils

import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from transformers import AutoConfig, AutoModel

import data_utils_NEW as data_utils
import my_features
from eval_utils import apply_heuristics

START_TAG = 7
STOP_TAG = 8

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self, bert_model_name, chunk_hidden_dim, max_chunk_len, max_seq_len,
                 feat_sz, batch_size, output_dim, use_features=False, bert_freeze=0):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.chunk_hidden_dim = chunk_hidden_dim
        self.max_chunk_len = max_chunk_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_dim = output_dim

        bert_config = AutoConfig.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)

        if bert_freeze > 0:
            # We freeze here the embeddings of the model
            for param in self.bert_model.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[:bert_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.use_features = use_features
        if not use_features:
            self.chunk_lstm = nn.LSTM(bert_config.hidden_size, chunk_hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        else:
            self.chunk_lstm = nn.LSTM(bert_config.hidden_size + feat_sz, chunk_hidden_dim, batch_first=True, bidirectional=True, num_layers=1)

        self.fc = nn.Linear(2 * chunk_hidden_dim, output_dim + 2)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(output_dim + 2, output_dim + 2))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()

    def load_transition_priors(self, transition_priors):
        with torch.no_grad():
            self.transitions.copy_(torch.from_numpy(transition_priors))

    def init_hidden_chunk_bilstm(self):
        # var1 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim)).cuda()
        # var2 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim)).cuda()
        var1 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim))
        var2 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.chunk_hidden_dim))
        return var1, var2

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.output_dim + 2), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas.cuda()
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.output_dim + 2):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.output_dim + 2)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score +  emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_bert_features(self, x, x_feats, x_len, x_chunk_len):
        self.bert_batch = 8
        #print("_get_bert_features()")
        #print("x", x.shape)

        input_ids = x[:,0,:,:]
        token_ids = x[:,1,:,:]
        attn_mask = x[:,2,:,:]

        max_seq_len = max(x_len)
        #print("x_len", x_len.shape, max_seq_len)
        #print("x_chunk_len", x_chunk_len.shape, x_chunk_len)

        # tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float().cuda()
        tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float()

        idx = 0
        for inp, tok, att, seq_length, chunk_lengths in zip(input_ids, token_ids, attn_mask, x_len, x_chunk_len):
            curr_max = max(chunk_lengths)

            inp = inp[:seq_length, :curr_max]
            tok = tok[:seq_length, :curr_max]
            att = att[:seq_length, :curr_max]
            #print("inp", inp.shape)

            # Run bert over this
            outputs = self.bert_model(inp, attention_mask=att, token_type_ids=tok,
                                      position_ids=None, head_mask=None)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            tensor_seq[idx, :seq_length] = pooled_output

            #print("output", pooled_output.shape)
        #print("tensor_seq.shape", tensor_seq.shape)

        # Now run the bilstm
        x_len = x_len.cpu()
        self.batch_size = x.shape[0]
        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()
        if self.use_features:
            x_feats = x_feats[:, :max_seq_len, :]
            tensor_seq = torch.cat((tensor_seq, x_feats), 2)
        tensor_seq = torch.nn.utils.rnn.pack_padded_sequence(tensor_seq, x_len, batch_first=True, enforce_sorted=False)
        tensor_seq, self.hidden_chunk_bilstm = self.chunk_lstm(tensor_seq, self.hidden_chunk_bilstm)
        # unpack
        tensor_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(tensor_seq, batch_first=True, total_length=max_seq_len)
        tensor_seq = self.fc(tensor_seq)
        '''
        chunk_output = []
        # partition per batch size so that it fits on mem
        for i in range(0, length, self.bert_batch):
            inp_batch = inp[i:i+self.bert_batch]
            tok_batch = tok[i:i+self.bert_batch]
            att_batch = att[i:i+self.bert_batch]
            #print("inp_batch", inp_batch.shape)

            # Run bert over this
            outputs = self.bert_model(inp_batch, attention_mask=att_batch, token_type_ids=tok_batch,
                                      position_ids=None, head_mask=None)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)

            print("output", pooled_output.shape)
            #chunk_output.append(pooled_output)

        print("----")
        print(len(chunk_output))
        '''
        return tensor_seq

    def _get_bilstm_features(self, x, x_feats, x_len, x_chunk_len):
        self.batch_size = x.shape[0]

        self.hidden_token_bilstm = self.init_hidden_token_bilstm()
        self.hidden_chunk_bilstm = self.init_hidden_chunk_bilstm()

        x = self.embedding(x)
        # unsqueeze all the chunks
        x = x.view(self.batch_size * self.max_chunk_len, self.max_seq_len, -1)
        # clamp everything to minimum length of 1, but keep the original variable to mask the output later
        x_chunk_len = x_chunk_len.view(-1)
        x_chunk_len_clamped = x_chunk_len.clamp(min=1, max=self.max_seq_len).cpu()

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_chunk_len_clamped, batch_first=True, enforce_sorted=False)
        x, self.hidden_token_bilstm = self.token_lstm(x, self.hidden_token_bilstm)
        # unpack
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # extract last timestep, since doing [-1] would get the padded zeros
        # idx = (x_chunk_len_clamped - 1).view(-1, 1).expand(x.size(0), x.size(2)).unsqueeze(1).cuda()
        idx = (x_chunk_len_clamped - 1).view(-1, 1).expand(x.size(0), x.size(2)).unsqueeze(1)
        x = x.gather(1, idx).squeeze()

        # revert back to (batch_size, max_chunk_len)
        x = x.view(self.batch_size, self.max_chunk_len, -1)
        if self.use_features:
            #print(x.shape, x_feats.shape)
            x = torch.cat((x, x_feats), 2)

        x_len = x_len.cpu()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, self.hidden_chunk_bilstm = self.chunk_lstm(x, self.hidden_chunk_bilstm)
        # unpack
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_chunk_len)
        x = self.fc(x)
        return x

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # score = torch.zeros(1).cuda()
        # temp = torch.tensor([START_TAG], dtype=torch.long).cuda()
        score = torch.zeros(1)
        temp = torch.tensor([START_TAG], dtype=torch.long)

        tags = torch.cat([temp, tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.output_dim + 2), -10000.)
        init_vvars[0][START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # forward_var = init_vvars.cuda()
        forward_var = init_vvars
        #print("feats.size()", feats.size())

        for feat_idx, feat in enumerate(feats):
            #print(feat_idx)
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.output_dim + 2):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, x, x_feats, x_len, x_chunk_len, y):
        # loss_accum = torch.autograd.Variable(torch.FloatTensor([0])).cuda()
        loss_accum = torch.autograd.Variable(torch.FloatTensor([0]))

        feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len)
        n = 0

        #print(x.size(), x_len.size(), feats.size(), y.size())
        #print(x_len)

        for x_len_i, sent_feats, tags in zip(x_len, feats, y):
            #print(sent_feats[:x_len_i].size(), tags[:x_len_i].size())
            forward_score = self._forward_alg(sent_feats[:x_len_i])
            gold_score = self._score_sentence(sent_feats[:x_len_i], tags[:x_len_i])
            loss_accum += forward_score - gold_score
            n += 1
        return loss_accum / n

    def predict_sequence(self, x, x_feats, x_len, x_chunk_len):
        # Get the emission scores from the BiLSTM
        #print("x.shape", x.shape)
        lstm_feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len)
        outputs = []
        for x_len_i, sequence in zip(x_len, lstm_feats):
            #print("sequence.shape", sequence[:x_len_i].shape)
            # Find the best path, given the features.
            score, tag_seq = self._viterbi_decode(sequence[:x_len_i])
            outputs.append(tag_seq)

        return outputs

    def forward(self, x, x_feats, x_len, x_chunk_len):  # dont confuse this with _forward_alg above
        output = self._get_bert_features(x, x_feats, x_len, x_chunk_len)
        return output

def evaluate_sequences(model, test_dataloader):
    model.eval()
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len in test_dataloader:
        # x = x.cuda()
        # x_feats = x_feats.cuda()
        # x_len = x_len.cuda()
        # x_chunk_len = x_chunk_len.cuda()
        # y = y.cuda()

        model.zero_grad()
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        # batch_y = y.view(-1)
        # Focus on non-pad elemens
        # idx = batch_y >= 0
        # batch_y = batch_y[idx]
        # label = batch_y.to('cpu').numpy()

        preds.append(list(batch_preds))
        # labels.append(list(label))
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs_emissions', action='store_true')
    parser.add_argument('--use_transition_priors', action='store_true')
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--printout', default=False, action='store_true')
    parser.add_argument('--features', default=False, action='store_true')
    parser.add_argument('--token_level', default=False, action='store_true', help='perform prediction at token level')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_embed_path', type=str)
    parser.add_argument('--word_embed_size', type=int, default=100)
    parser.add_argument('--token_hidden_dim', type=int, default=50)
    parser.add_argument('--chunk_hidden_dim', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--write_results', default=False, action='store_true')
    parser.add_argument('--heuristics', default=False, action='store_true')
    parser.add_argument('--bert_model', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--cuda_device', type=int, default=0)

    # I am not sure about what this is anymore
    parser.add_argument('--partition_sentence', default=False, action='store_true')
    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    # if args.protocol not in protocols:
    #     print("Specify a valid protocol")
    #     exit(-1)

    # if args.cuda_device >= 0:
    #     is_cuda_avail = torch.cuda.is_available()
    #     if not is_cuda_avail:
    #         print("ERROR: There is no CUDA device available, you need a GPU to train this model.")
    #         exit(-1)
    #     elif args.cuda_device >= torch.cuda.device_count():
    #         print("ERROR: Please specify a valid cuda device, you have {} devices".format(torch.cuda.device_count()))
    #         exit(-1)
    #     torch.cuda.set_device('cuda:{}'.format(args.cuda_device))
    #     torch.backends.cudnn.benchmark=True
    # else:
    #     print("ERROR: You need a GPU to train this model. Please specify a valid cuda device, you have {} devices".format(torch.cuda.device_count()))
    #     exit(-1)

    # args.savedir_fold = os.path.join(args.savedir, "checkpoint_{}.pt".format(args.protocol))
    args.savedir_fold = os.path.join(args.savedir, "checkpoint_TCP.pt")

    word2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}

    # Get variable and state definitions
    def_vars = set(); def_states = set(); def_events = set(); def_events_constrained = set()
    data_utils.get_definitions(def_vars, def_states, def_events_constrained, def_events)

    args.pred = ["nlp-parser/predict/rfc/{}_annotation_9.txt".format(args.protocol)]

    pred_data_orig, level_h, level_d = my_data_utils.get_data(args.pred, word2id, id2word, id2cap)

    max_chunks, max_tokens = my_data_utils.max_lengths(pred_data_orig)
    print(max_chunks, max_tokens)

    vocab_size = 0; pos_size = 0; def_var_ids = []; def_state_ids = []; def_event_ids = []
    pred_data_feats = my_features.transform_features(pred_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, True)

    pred_data, pre_len, pre_chunk_len = my_data_utils.bert_sequences(pred_data_orig, max_chunks, max_tokens, id2word, args.bert_model)


    pred_data_feats = data_utils.pad_features(pred_data_feats, max_chunks)

    feat_sz = pred_data_feats.shape[2]

    pred_dataset = my_data_utils.ChunkDataset(pred_data, pred_data_feats, pre_len, pre_chunk_len)

    pred_dataloder = torch.utils.data.DataLoader(pred_dataset, batch_size=args.batch_size, shuffle=False)

    id2tag = {0: 'B-TRIGGER', 1: 'B-ACTION', 2: 'O', 3: 'B-TRANSITION', 4: 'B-TIMER', 5: 'B-ERROR', 6: 'B-VARIABLE'}
    tag2id = {'B-TRIGGER': 0, 'B-ACTION': 1, 'O': 2, 'B-TRANSITION': 3, 'B-TIMER': 4, 'B-ERROR': 5, 'B-VARIABLE': 6}
    classes = [0, 1, 2, 3, 4, 5, 6]

    # Create model
    model = BERT_BiLSTM_CRF(args.bert_model,
                            args.chunk_hidden_dim,
                            max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(tag2id),
                            use_features=args.features, bert_freeze=10)
    # model.cuda()

    if args.do_eval:
        # Load model
        model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        # y_test, y_pred = evaluate_sequences(model, test_dataloader)
        y_pred = evaluate_sequences(model, pred_dataloder)

        # y_test_trans = data_utils.translate(y_test, id2tag)
        y_pred_trans = data_utils.translate(y_pred, id2tag)

        pred_data_alt, y_pred_trans_alt, level_h_alt, level_d_alt = data_utils.alternative_expand(pred_data_orig, y_pred_trans, level_h, level_d, id2word, debug=True)

        def_states_protocol = {}; def_events_protocol = {}; def_events_constrained_protocol = {}; def_variables_protocol = {}
        data_utils.get_protocol_definitions(args.protocol, def_states_protocol, def_events_constrained_protocol, def_events_protocol, def_variables_protocol)

        y_pred_trans_alt = \
            apply_heuristics(pred_data_alt, y_pred_trans_alt, y_pred_trans_alt,
                             level_h_alt, level_d_alt,
                             id2word, def_states_protocol, def_events_protocol, def_variables_protocol,
                             transitions=args.heuristics, outside=args.heuristics, actions=args.heuristics,
                             consecutive_trans=True)

        pred_data_orig, y_pred_trans, level_h_trans, level_d_trans = \
            data_utils.alternative_join(
                pred_data_alt, y_pred_trans_alt,
                level_h_alt, level_d_alt,
                id2word, debug=True)
        #
        # if args.heuristics:
        #     _, y_test_trans_eval = data_utils.expand(X_test_data_old, y_test_trans, id2word, debug=False)
        #     evaluate(y_test_trans_eval, y_pred_trans)

        if args.write_results:
            output_xml = os.path.join(args.outdir, "{}.xml".format(args.protocol))
            results = my_data_utils.write_results(pred_data_orig, y_pred_trans, y_pred_trans, level_h_trans, level_d_trans,
                                               id2word, def_states_protocol, def_events_protocol, def_events_constrained_protocol,
                                               args.protocol, cuda_device=args.cuda_device)
            with open(output_xml, "w") as fp:
                fp.write(results)

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(4321)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(4321)
    random.seed(4321)

    main()
