import dgl
import torch
#from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
from tokenizers.pre_tokenizers import CharDelimiterSplit
import re
import sys
import os
device = torch.device("cuda")

bert_path = "/gpfs/u/home/DLTM/DLTMboxi/scratch/env/finbert-pretrain/"

#config = AutoConfig.from_pretrained(bert_path, output_hidden_states = True) 

model = BertModel.from_pretrained(bert_path,output_hidden_states = True)

tokenizer = BertTokenizer.from_pretrained(bert_path)

tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

model = model.to(device)
model.eval()
#tokenizer = tokenizer.to(device)
#tokenizer.sep_token = ' '

class Node:
    def __init__(self,word,entity,embedding,doc_id):
        self.word = word
        self.entity = entity
        self.embedding = embedding
        self.doc_id = doc_id
    def __str__(self):
        return self.entity+"; "+self.word+";"+str(self.doc_id)

def sentence_to_bert_embeddings(sentence):
    #print(sentence)
    # Add the special tokens.
    marked_text = "[CLS] " + sentence + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    
    if len(tokenized_text) > 512:
        #print("more than 512!")
        track_file.write("more than 512!\n")
        track_file.write("len: "+str(len(tokenized_text)))
        track_file.flush()
        #print(len(tokenized_text))
        tokenized_text = tokenizer.tokenize(sentence)
        #print(tokenized_text)
        #print(len(tokenized_text))
        #for i in range(len(tokenized_text)):
            #print(tokenized_text[i])
        tokens = tokenizer.encode_plus(tokenized_text, add_special_tokens=False, return_tensors='pt')
        input_id_chunks = list(tokens['input_ids'][0].split(510))
        mask_chunks = list(tokens['attention_mask'][0].split(510))
        tokenized_text_list = [['[CLS]']+tokenized_text[:510]+['[SEP]'],['[CLS]']+tokenized_text[510:]+['[SEP]']+(510-len(tokenized_text[510:]))*['[PAD]']]
        #print('len [0]: ',len(tokenized_text_list[0]))
        #print('len [1]: ',len(tokenized_text_list[1]))
        #print('len input chunks:',len(input_id_chunks))
        #print(amber)
        #print(tokenizer.cls_token_id)
        for i in range(len(input_id_chunks)):
            cls_tok = torch.Tensor([tokenizer.cls_token_id])
            #pad_tok = torch.Tensor([tokenizer.pad_token_id])
            sep_tok = torch.Tensor([tokenizer.sep_token_id])
            mask_tok = torch.Tensor([1])
            # add CLS and SEP tokens to input IDs
            input_id_chunks[i] = torch.cat([cls_tok, input_id_chunks[i], sep_tok])
            # add attention tokens to attention mask
            mask_chunks[i] = torch.cat((mask_tok, mask_chunks[i], mask_tok))
            # get required padding length
            pad_len = 512 - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat((input_id_chunks[i], torch.Tensor([tokenizer.pad_token_id] * pad_len)))
                mask_chunks[i] = torch.cat((mask_chunks[i], torch.Tensor([tokenizer.pad_token_id] * pad_len)))
        #for chunk in input_id_chunks:
            #print(len(chunk))
        #print(amber)
        input_ids = torch.stack(input_id_chunks).to(device)
        attention_mask = torch.stack(mask_chunks).to(device)

        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int()
        }

        model.eval()
        #print(tokens_tensor.shape)
        #print(segments_tensors.shape)
        with torch.no_grad():
            outputs = model(**input_dict)
            hidden_states = outputs[2]
        #print(hidden_states)
        #print(hidden_states[0].shape)
        #print(hidden_states[1].shape)
        #print(len(hidden_states))
        #del input_ids
        #del attention_mask
        #outputs.detach().cpu()
        #hidden_states.detach().cpu()
        token_embeddings = torch.stack(hidden_states, dim=0)
        del outputs
        del hidden_states
        #print(token_embeddings.size())

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        #print(token_embeddings.size())

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,2,0,3)

        #print(token_embeddings.size())

        final_results = []
        for i in range(token_embeddings.shape[0]):
            
            #print("i="+str(i))
            token_vecs_sum = []

            # For each token in the sentence...
            for token in token_embeddings[i]:

                # `token` is a [12 x 768] tensor

                # Sum the vectors from the last four layers.
                sum_vec = torch.sum(token[-4:], dim=0)
                
                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(sum_vec)
            count = 0
            results = []
            previous_vector = None
            #previous_word_token = None
            #current_vector = None
            vectors_to_mean = []
            #print(len(token_vecs_sum))
            while(True):
                #print(count)
                if count == len(token_vecs_sum):
                    break
                #print('len: ',len(token_vecs_sum),' count: ',count)
                current_vector = token_vecs_sum[count].unsqueeze(0)
                #print('len: ',len(tokenized_text),' count: ',count)
                current_word_token = tokenized_text_list[i][count]
                if current_word_token[:2] == "##":
                    vectors_to_mean.append(previous_vector)
                    vectors_to_mean.append(current_vector)
                    increment = 1
                    while(True):
                        next_word_token = tokenized_text_list[i][count+increment]
                        next_vector = token_vecs_sum[count+increment].unsqueeze(0)
                        if next_word_token[:2] == "##":
                            vectors_to_mean.append(next_vector)
                            increment += 1
                            continue
                        else:
                            count += increment
                            #print(count)
                            break
                    vectors_to_mean = torch.cat(vectors_to_mean,0)
                    #for vec in vectors_to_mean:  
                        #print(vec[:5])
                    current_vector = torch.mean(vectors_to_mean,0).unsqueeze(0)
                    #print(current_vector[:5])
                    vectors_to_mean = []
                    results.pop()
                    results.append(current_vector)
                    #print('meaned vector size: '+str(current_vector.shape))
                    continue
                results.append(current_vector)
                #print("non-meaned current vector size: "+str(current_vector.shape))
                previous_vector = current_vector
                count += 1
            results = torch.cat(results,0)
            final_results.append(results)
        #print(len(sentence.split()))
        #print(final_results[0].shape)
        #print(final_results[1].shape)
        del input_dict
        del input_ids
        del attention_mask

        final_result = torch.cat((final_results[0][:-1],final_results[1][1:]))[:len(sentence.split())+2]
        #print(amber)
        #print(final_result.shape)
        return final_result
        #print(final_result.shape)
        #print(amber)
    else:
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Display the words with their indeces.
        #for tup in zip(tokenized_text, indexed_tokens):
            #print('{:<12} {:>6,}'.format(tup[0], tup[1]))
        
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        model.eval()
        #print(tokens_tensor.shape)
        #print(segments_tensors.shape)
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        #outputs.detach().cpu()
        #hidden_states.detach().cpu()
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)
        del outputs
        del hidden_states
        #print(token_embeddings.size())

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        #print(token_embeddings.size())

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        #print(token_embeddings.size())

        token_vecs_sum = []

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        #print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
        #print(token_vecs_sum[1].shape)
        count = 0
        results = []
        previous_vector = None
        #previous_word_token = None
        #current_vector = None
        vectors_to_mean = []
        while(True):
            current_vector = token_vecs_sum[count].unsqueeze(0)
            
            current_word_token = tokenized_text[count]
            if current_word_token[:2] == "##":
                vectors_to_mean.append(previous_vector)
                vectors_to_mean.append(current_vector)
                increment = 1
                while(True):
                    next_word_token = tokenized_text[count+increment]
                    next_vector = token_vecs_sum[count+increment].unsqueeze(0)
                    if next_word_token[:2] == "##":
                        vectors_to_mean.append(next_vector)
                        increment += 1
                        continue
                    else:
                        count += increment
                        break
                vectors_to_mean = torch.cat(vectors_to_mean,0)
                #for vec in vectors_to_mean:  
                    #print(vec[:5])
                current_vector = torch.mean(vectors_to_mean,0).unsqueeze(0)
                #print(current_vector[:5])
                vectors_to_mean = []
                results.pop()
                results.append(current_vector)
                #print('meaned vector size: '+str(current_vector.shape))
                continue
            
            results.append(current_vector)
            #print("non-meaned current vector size: "+str(current_vector.shape))
            previous_vector = current_vector
            count += 1
            if count == len(token_vecs_sum):
                break
        results = torch.cat(results,0)
        tokens_tensor.detach().cpu()
        segments_tensors.detach().cpu()
        del tokens_tensor
        del segments_tensors
        return results


sec_fname = "../punkt-truly-all-amrs/"
#graphs_list = []
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])
try: 
    os.mkdir('truly-all-results-graphs-hk-finbert-plan-D')
except OSError as error: 
    print(error)
track_file = open('tracking_file_all-results_construct_hk-finbert_plan_D_'+str(start_id)+"-"+str(end_id)+".txt",'a+')
with open('../truly-all-amrs-file-names.txt','r') as f:
    filenames = f.readlines()

for i in range(start_id, end_id):
    #print("i = "+str(i))
    filename = filenames[i].strip()
    track_file.write("i = "+str(i)+'\n')
    track_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
    track_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_allocated()/1024/1024))+' mem allocated\n')
    
    #torch.cuda.empty_cache()
    #print(torch.cuda.memory_allocated())
    #print(torch.cuda.memory_reserved())
    fname = sec_fname+filename
    ffile = open(fname,'r')
    flines = ffile.readlines()

    current_snt = ''
    current_snt_embeddings = None
    current_snt_cls = None
    previous_snt_cls = None
    current_snt_nodes_dict = {}

    count = 0
    doc_nodes_list = []
    node_index = 0
    doc_nodes_list.append(Node("doc_cls","doc_cls",torch.zeros(768),node_index))
    node_index += 1

    edge_list1 = []
    edge_list2 = []
    while(True):
        line = flines[count]
        if line[:7] == "# ::tok":
            #edges from the PREVIOUS snt_cls to each node in PREVIOUS snt happen here
            #also edge from doc_cls to PREVIOUS snt_cls and every node in PREVIOUS snt happens here
            if current_snt_cls is not None:
                for key in current_snt_nodes_dict:
                    #connect PREVIOUS snt_cls to a node in PREVIOUS snt
                    n1 = current_snt_cls.doc_id
                    n2 = current_snt_nodes_dict[key].doc_id
                    edge_list1.append(n1)
                    edge_list2.append(n2)
                    #connect doc_cls to a node in PREVIOUS snt
                    n1 = 0
                    n2 = current_snt_nodes_dict[key].doc_id
                    edge_list1.append(n1)
                    edge_list2.append(n2)
                n1 = 0
                n2 = current_snt_cls.doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)


            previous_snt_cls = current_snt_cls
            #clear previous current_snt_nodes_dict
            current_snt_nodes_dict.clear()

            current_snt = re.sub(r'(?<=\S)[^a-zA-Z0-9\s]','',line[7:])
            current_snt = re.sub(r'[^a-zA-Z0-9\s](?=\S)','',current_snt)
            #current_snt = current_snt.replace('.','')+'.'
            #print(current_snt)
            current_snt_embeddings = sentence_to_bert_embeddings(current_snt)[1:-1].cpu()
            current_snt = current_snt.split()
            #print(current_snt_embeddings.shape[0])
            #print(len(current_snt))
            #if not len(current_snt) == current_snt_embeddings.shape[0]:
                #print(line[7:])
            assert len(current_snt) == current_snt_embeddings.shape[0]
            #current_snt_embeddings = global_vectors.get_vecs_by_tokens(current_snt,lower_case_backup=True)
            #print(len(current_snt))
            #print(current_snt_embeddings.shape)
            current_snt_cls = Node("snt_cls","snt_cls",torch.zeros(768),node_index)
            node_index += 1
            doc_nodes_list.append(current_snt_cls)
            #connect previous snt_cls t current snt_cls
            if previous_snt_cls is not None:
                n1 = previous_snt_cls.doc_id
                n2 = current_snt_cls.doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)
            count += 1
            continue
        elif line[:8] == "# ::node":
            node_info_list = line[8:].split()
            snt_node_id = node_info_list[0]
            word_index = int(node_info_list[-1].split('-')[0])
            word = current_snt[word_index]
            entity = node_info_list[1]
            embedding = current_snt_embeddings[word_index]
            node = Node(word, entity, embedding, node_index)
            node_index += 1
            doc_nodes_list.append(node)
            current_snt_nodes_dict[snt_node_id] = node
            count += 1
            continue
        elif line[:8] == "# ::edge":
            edge_info_list = line[8:].split()
            key1 = edge_info_list[-2]
            key2 = edge_info_list[-1]
            n1 = current_snt_nodes_dict[key1].doc_id
            n2 = current_snt_nodes_dict[key2].doc_id
            edge_list1.append(n1)
            edge_list2.append(n2)
            count += 1
            continue
        
        count += 1
        if count == len(flines):
            for key in current_snt_nodes_dict:
                #connect PREVIOUS snt_cls to a node in PREVIOUS snt
                n1 = current_snt_cls.doc_id
                n2 = current_snt_nodes_dict[key].doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)
                #connect doc_cls to a node in PREVIOUS snt
                n1 = 0
                n2 = current_snt_nodes_dict[key].doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)
            n1 = 0
            n2 = current_snt_cls.doc_id
            edge_list1.append(n1)
            edge_list2.append(n2)
            break

    #print(len(doc_nodes_list))
    #print(amber)
    #doc_nodes_list = doc_nodes_list.to(device)
    num_of_nodes = len(doc_nodes_list)
    g = dgl.graph((edge_list1,edge_list2),num_nodes=num_of_nodes).to(device)
    #g = g.cpu()
    g.ndata['h'] = torch.zeros((num_of_nodes, 768)).to(device)
    for l in range(num_of_nodes):
        g.nodes[l].data['h'] = doc_nodes_list[l].embedding[None,:].to(device)
        doc_nodes_list[l].embedding = doc_nodes_list[l].embedding.detach().cpu()
        del doc_nodes_list[l].embedding
    del doc_nodes_list
    g = g.cpu()
    g.ndata['h'] = g.ndata['h'].cpu()
    final_graph = dgl.to_bidirected(g,copy_ndata=True).cpu()
    del g.ndata['h']
    del g
    
    #graphs_list.append(final_graph)
    #torch.cuda.reset_max_memory_allocated() 
    torch.cuda.empty_cache()
    dgl.save_graphs("truly-all-results-graphs-hk-finbert-plan-D/"+filename[:-3]+"graph",[final_graph])

    #torch.cuda.reset_max_memory_allocated() 
    #print(torch.cuda.memory_allocated()/1024/1024)
    #print(torch.cuda.memory_reserved()/1024/1024)
    track_file.write("doc num: "+str(i)+" after train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
    track_file.write("doc num: "+str(i)+" after train: "+str(int(torch.cuda.memory_allocated()/1024/1024))+' mem allocated\n')
    track_file.flush()
track_file.close()
#print(len(graphs_list))
