from GTN import *

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    num_node_features = 10 
    num_classes = 2 
    num_edge_features=11
    model = TransformerGNNNet(num_node_features, num_edge_features, num_classes)
    model = model.to(device)
    
    model_load_path = 'model/GTN_model.pth'
    lane_save_path = 'result/predicted_lane.csv'
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    df_edge_test=pd.read_csv("processed_data/edges_test.csv")
    df_node_test=pd.read_csv("processed_data/nodes_test.csv")
    
    columns_to_normalize = ['hsdf_dis' ,'dis0','dis1','dis2','dis3','dis4','dis5','dis6','dis7','dis8','dis9']
    df_edge_test_normalized = exp_norm_replace(df_edge_test.copy(), columns_to_normalize, 2)
    test_graphs,test_node_ids=load_data(df_edge_test_normalized,df_node_test)
    inter_test_ids=[9,39,40,52]
    test_selected_node_ids_list= np.array([], dtype=int)
    for i in range(0,len(test_graphs)):
        prediction = model(test_graphs[i].to(device))
        result=F.softmax(prediction, dim=1)
        index_list = np.where(result[:, 1].cpu() > 0.7)[0]
        selected_node_ids = test_node_ids[inter_test_ids[i]][index_list]
        test_selected_node_ids_list=np.append(test_selected_node_ids_list,selected_node_ids)
    
    traj_grouped=pd.read_csv('data/traj.csv')
    test_filtered_df = traj_grouped[traj_grouped['trajectory_id'].isin(test_selected_node_ids_list)]
    
    gdf_test = gpd.GeoDataFrame(columns=['inter_id', 'trajectory_id', 'geometry'])
    grouped = test_filtered_df.groupby('trajectory_id')
    for name, group in grouped:
        line = LineString(zip(group['x'], group['y']))
        inter_id = group['inter_id'].iloc[0]
        new_row = pd.DataFrame({'inter_id': [inter_id], 'trajectory_id': [name], 'geometry': [line]})
        gdf_test = pd.concat([gdf_test, gpd.GeoDataFrame(new_row)], ignore_index=True)
    gdf_test.to_csv(lane_save_path,index=False)
    print("Predicted lane has been saved in",lane_save_path)
