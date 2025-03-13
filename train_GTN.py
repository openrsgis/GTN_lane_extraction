from GTN import *

if __name__ == '__main__':
    set_seed(7)
    # read_data
    df_node_train=pd.read_csv("processed_data/nodes_train.csv")
    df_edge_train=pd.read_csv("processed_data/edges_train.csv")
    df_node_train_augmented=pd.read_csv("processed_data/nodes_train_augmented.csv")
    df_edge_train_augmented=pd.read_csv("processed_data/edges_train_augmented.csv")
    
    columns_to_normalize = ['hsdf_dis' ,'dis0','dis1','dis2','dis3','dis4','dis5','dis6','dis7','dis8','dis9']
    df_edge_train_normalized = exp_norm_replace(df_edge_train_augmented.copy(), columns_to_normalize, 2)
    
    train_graphs,train_node_ids=load_data(df_edge_train_normalized,df_node_train_augmented)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    D_hsdf_train_list = np.load('processed_data/D_hsdf_train_augmented.npz') 
    
    D_hsdf_train_list = [D_hsdf_train_list[f'arr_{i}'] for i in range(len(D_hsdf_train_list))]
    D_norm_train_list= normalize_D(D_hsdf_train_list,device)
    
    
    # model training
    num_node_features = 10 
    num_classes = 2 
    num_edge_features=11
    model = TransformerGNNNet(num_node_features,num_edge_features, num_classes)
    model = model.to(device)
    torch.cuda.empty_cache()
    
    from torch_geometric.data import Batch
    from IPython.display import clear_output
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.0001)
    
    loss_values = []
    
    for epoch in range(400): 
        model.train() 
        optimizer.zero_grad()
        batch = Batch.from_data_list(train_graphs).to(device)
        out = model(batch)
        matcher = HungarianMatcher()
    
        loss = get_loss_batch(matcher,batch, out, D_norm_train_list) 
        loss.backward()
        optimizer.step()  
        average_loss = loss.item()
        loss_values.append(average_loss)
        print("epoch:",epoch,"  loss:",average_loss)
        
    print('Finish training.')    
    # save model
    model_save_path = 'model/GTN_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print('Model has been saved to', model_save_path)
