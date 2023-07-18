import dgl
import pandas as pd
'''
df_train = pd.read_csv('../new-tech-2010-to-2018-result.csv')

for plan in ['C','D','E']:
    parent_dir = f'truly-all-results-graphs-hk-finbert-plan-{plan}'
    graph_list = []
    for index, row in df_train.iterrows():
        print(f'{plan}-{index}')
        graph_path = parent_dir+f'/{row["ticker_and_date"]}.graph'
        graph_list.append(dgl.load_graphs(graph_path)[0][0])
    dgl.save_graphs(f'tech-2010-to-2018-plan-{plan}.graphs',graph_list)
'''

df_train = pd.read_csv('../new-tech-2019-result.csv')

for plan in ['C','D','E']:
    parent_dir = f'truly-all-results-graphs-hk-finbert-plan-{plan}'
    graph_list = []
    for index, row in df_train.iterrows():
        print(f'{plan}-{index}')
        graph_path = parent_dir+f'/{row["ticker_and_date"]}.graph'
        graph_list.append(dgl.load_graphs(graph_path)[0][0])
    dgl.save_graphs(f'tech-2019-plan-{plan}.graphs',graph_list)