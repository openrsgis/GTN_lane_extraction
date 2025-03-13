from data_processing import *


if __name__ == '__main__':
    traj_grouped=pd.read_csv('data\traj.csv')
    traj_grouped_projected=trans_crs(traj_grouped)
    df_edge=compute_distances(traj_grouped_projected)
    df_edge.to_csv('processed_data\edge.csv',index=False)
    trueLine_grouped=pd.read_csv('data\trueLane.csv')
    trueLine_grouped['geometry'] = trueLine_grouped['geometry'].apply(wkt.loads)
    trueLine_add_sampling=resample_lines(trueLine_grouped, 0.5/111111)
    train_inter_ids=[0, 10,12,26, 28, 37, 43, 44, 48, 51, 61,67,73, 92, 122,123] # train inter id
    test_inter_ids=[9,39,40,52] # test inter id
    
    D_hsdf_train_list=[]
    for inter_id in tqdm(train_inter_ids):
        df_traj_test=traj_grouped[traj_grouped['inter_id']==inter_id]
        df_traj_test = df_traj_test.sort_values(by='trajectory_id')
        trueLine_test=trueLine_add_sampling[trueLine_add_sampling['inter_id']==inter_id]
        trajs_test=df_coordinate_projection(df_traj_test)
        line_test=line_coordinate_projection(trueLine_test)

        D_hsdf= hausdorffMatrix(trajs_test, line_test)
        D_hsdf_train_list.append(D_hsdf)

    D_hsdf_test_list=[]
    for inter_id in tqdm(test_inter_ids):
        df_traj_test=traj_grouped[traj_grouped['inter_id']==inter_id]
        df_traj_test = df_traj_test.sort_values(by='trajectory_id')
        trueLine_test=trueLine_add_sampling[trueLine_add_sampling['inter_id']==inter_id]
        trajs_test=df_coordinate_projection(df_traj_test)
        line_test=line_coordinate_projection(trueLine_test)
        D_hsdf= hausdorffMatrix(trajs_test, line_test)
        D_hsdf_test_list.append(D_hsdf)

    np.savez('processed_data\D_hsdf_train.npz', *D_hsdf_train_list)
    np.savez('processed_data\D_hsdf_test.npz', *D_hsdf_test_list)
    df_edge_train=df_edge[df_edge['inter_id'].isin(train_inter_ids)]
    df_edge_test=df_edge[df_edge['inter_id'].isin(test_inter_ids)]
    df_edge_train.to_csv('processed_data\edges_train.csv',index=False)
    df_edge_test.to_csv('processed_data\edges_test.csv',index=False)

    traj_grouped_projected=trans_crs(traj_grouped)
    traj_norm=normalize_trajectory_data(traj_grouped_projected)
    df_node=get_node_df(traj_norm)
    df_node_train=df_node[df_node['inter_id'].isin(train_inter_ids)]
    df_node_test=df_node[df_node['inter_id'].isin(test_inter_ids)]
    df_node_train.to_csv('processed_data\nodes_train.csv',index=False)
    df_node_test.to_csv('processed_data\nodes_test.csv',index=False)


    # data augmenting
    index_list0,selected_traj_ids_list0=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.0,0) # different sampling rate
    index_list1,selected_traj_ids_list1=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.1,1)
    index_list2,selected_traj_ids_list2=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.2,2)
    index_list3,selected_traj_ids_list3=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.3,3)
    index_list4,selected_traj_ids_list4=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.4,4)
    index_list5,selected_traj_ids_list5=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.5,5)
    index_list6,selected_traj_ids_list6=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.6,6)
    index_list7,selected_traj_ids_list7=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.7,7)
    index_list8,selected_traj_ids_list8=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.8,8)
    index_list9,selected_traj_ids_list9=get_subgraph_index(D_hsdf_train_list,traj_grouped,train_inter_ids,0.9,9)

    mask_edge0 = df_edge_train['traj1'].isin(selected_traj_ids_list0) & df_edge_train['traj2'].isin(selected_traj_ids_list0)
    sub_edge_df0 = df_edge_train[mask_edge0].copy()
    sub_edge_df0['inter_id'] += 200
    sub_edge_df0['traj1'] += 0.1
    sub_edge_df0['traj2'] += 0.1
    mask_edge1 = df_edge_train['traj1'].isin(selected_traj_ids_list1) & df_edge_train['traj2'].isin(selected_traj_ids_list1)
    sub_edge_df1 = df_edge_train[mask_edge1].copy()
    sub_edge_df1['inter_id'] += 400
    sub_edge_df1['traj1'] += 0.2
    sub_edge_df1['traj2'] += 0.2
    mask_edge2 = df_edge_train['traj1'].isin(selected_traj_ids_list2) & df_edge_train['traj2'].isin(selected_traj_ids_list2)
    sub_edge_df2 = df_edge_train[mask_edge2].copy()
    sub_edge_df2['inter_id'] += 600
    sub_edge_df2['traj1'] += 0.3
    sub_edge_df2['traj2'] += 0.3
    mask_edge3 = df_edge_train['traj1'].isin(selected_traj_ids_list3) & df_edge_train['traj2'].isin(selected_traj_ids_list3)
    sub_edge_df3 = df_edge_train[mask_edge3].copy()
    sub_edge_df3['inter_id'] += 800
    sub_edge_df3['traj1'] += 0.4
    sub_edge_df3['traj2'] += 0.4
    mask_edge4 = df_edge_train['traj1'].isin(selected_traj_ids_list4) & df_edge_train['traj2'].isin(selected_traj_ids_list4)
    sub_edge_df4 = df_edge_train[mask_edge4].copy()
    sub_edge_df4['inter_id'] += 1000
    sub_edge_df4['traj1'] += 0.5
    sub_edge_df4['traj2'] += 0.5
    mask_edge5 = df_edge_train['traj1'].isin(selected_traj_ids_list5) & df_edge_train['traj2'].isin(selected_traj_ids_list5)
    sub_edge_df5 = df_edge_train[mask_edge5].copy()
    sub_edge_df5['inter_id'] += 1200
    sub_edge_df5['traj1'] += 0.6
    sub_edge_df5['traj2'] += 0.6
    mask_edge6 = df_edge_train['traj1'].isin(selected_traj_ids_list6) & df_edge_train['traj2'].isin(selected_traj_ids_list6)
    sub_edge_df6 = df_edge_train[mask_edge6].copy()
    sub_edge_df6['inter_id'] += 1400
    sub_edge_df6['traj1'] += 0.7
    sub_edge_df6['traj2'] += 0.7
    mask_edge7 = df_edge_train['traj1'].isin(selected_traj_ids_list7) & df_edge_train['traj2'].isin(selected_traj_ids_list7)
    sub_edge_df7 = df_edge_train[mask_edge7].copy()
    sub_edge_df7['inter_id'] += 1600
    sub_edge_df7['traj1'] += 0.8
    sub_edge_df7['traj2'] += 0.8
    mask_edge8 = df_edge_train['traj1'].isin(selected_traj_ids_list8) & df_edge_train['traj2'].isin(selected_traj_ids_list8)
    sub_edge_df8 = df_edge_train[mask_edge8].copy()
    sub_edge_df8['inter_id'] += 1800
    sub_edge_df8['traj1'] += 0.9
    sub_edge_df8['traj2'] += 0.9
    mask_edge9 = df_edge_train['traj1'].isin(selected_traj_ids_list9) & df_edge_train['traj2'].isin(selected_traj_ids_list9)
    sub_edge_df9 = df_edge_train[mask_edge9].copy()
    sub_edge_df9['inter_id'] += 2000
    sub_edge_df9['traj1'] += 0.95
    sub_edge_df9['traj2'] += 0.95
    df_edge_train_augmented=pd.concat([df_edge_train, sub_edge_df0,sub_edge_df1,sub_edge_df2,sub_edge_df3,sub_edge_df4,
                                       sub_edge_df5,sub_edge_df6,sub_edge_df7,sub_edge_df8,sub_edge_df9], ignore_index=True)

    mask_node0 = df_node_train['traj_id'].isin(selected_traj_ids_list0) 
    sub_node_df0 = df_node_train[mask_node0].copy()
    sub_node_df0['inter_id'] += 200
    sub_node_df0['traj_id'] += 0.1

    mask_node1 = df_node_train['traj_id'].isin(selected_traj_ids_list1)
    sub_node_df1 = df_node_train[mask_node1].copy()
    sub_node_df1['inter_id'] += 400
    sub_node_df1['traj_id'] += 0.2

    mask_node2 = df_node_train['traj_id'].isin(selected_traj_ids_list2) 
    sub_node_df2 = df_node_train[mask_node2].copy()
    sub_node_df2['inter_id'] +=600
    sub_node_df2['traj_id'] += 0.3

    mask_node3 = df_node_train['traj_id'].isin(selected_traj_ids_list3)
    sub_node_df3 = df_node_train[mask_node3].copy()
    sub_node_df3['inter_id'] += 800
    sub_node_df3['traj_id'] += 0.4

    mask_node4 = df_node_train['traj_id'].isin(selected_traj_ids_list4)
    sub_node_df4 = df_node_train[mask_node4].copy()
    sub_node_df4['inter_id'] += 1000
    sub_node_df4['traj_id'] += 0.5

    mask_node5 = df_node_train['traj_id'].isin(selected_traj_ids_list5)
    sub_node_df5 = df_node_train[mask_node5].copy()
    sub_node_df5['inter_id'] += 1200
    sub_node_df5['traj_id'] += 0.6

    mask_node6 = df_node_train['traj_id'].isin(selected_traj_ids_list6)
    sub_node_df6 = df_node_train[mask_node6].copy()
    sub_node_df6['inter_id'] += 1400
    sub_node_df6['traj_id'] += 0.7

    mask_node7 = df_node_train['traj_id'].isin(selected_traj_ids_list7)
    sub_node_df7 = df_node_train[mask_node7].copy()
    sub_node_df7['inter_id'] += 1600
    sub_node_df7['traj_id'] += 0.8

    mask_node8 = df_node_train['traj_id'].isin(selected_traj_ids_list8)
    sub_node_df8 = df_node_train[mask_node8].copy()
    sub_node_df8['inter_id'] += 1800
    sub_node_df8['traj_id'] += 0.9

    mask_node9 = df_node_train['traj_id'].isin(selected_traj_ids_list9)
    sub_node_df9 = df_node_train[mask_node9].copy()
    sub_node_df9['inter_id'] += 2000
    sub_node_df9['traj_id'] += 0.95
    df_node_train_augmented=pd.concat([df_node_train, sub_node_df0,sub_node_df1,sub_node_df2,sub_node_df3,sub_node_df4,
                                      sub_node_df5,sub_node_df6,sub_node_df7,sub_node_df8,sub_node_df9], ignore_index=True)


    D_hsdf_train_list0=[]
    D_hsdf_train_list1=[]
    D_hsdf_train_list2=[]
    D_hsdf_train_list3=[]
    D_hsdf_train_list4=[]
    D_hsdf_train_list5=[]
    D_hsdf_train_list6=[]
    D_hsdf_train_list7=[]
    D_hsdf_train_list8=[]
    D_hsdf_train_list9=[]
    for i in range(0,len(D_hsdf_train_list)):
        D_hsdf_train_list0.append(D_hsdf_train_list[i][index_list0[i]])
        D_hsdf_train_list1.append(D_hsdf_train_list[i][index_list1[i]])
        D_hsdf_train_list2.append(D_hsdf_train_list[i][index_list2[i]])
        D_hsdf_train_list3.append(D_hsdf_train_list[i][index_list3[i]])
        D_hsdf_train_list4.append(D_hsdf_train_list[i][index_list4[i]])
        D_hsdf_train_list5.append(D_hsdf_train_list[i][index_list5[i]])
        D_hsdf_train_list6.append(D_hsdf_train_list[i][index_list6[i]])
        D_hsdf_train_list7.append(D_hsdf_train_list[i][index_list7[i]])
        D_hsdf_train_list8.append(D_hsdf_train_list[i][index_list8[i]])
        D_hsdf_train_list9.append(D_hsdf_train_list[i][index_list9[i]])
    D_hsdf_train_list_augmented=(D_hsdf_train_list + D_hsdf_train_list0 + D_hsdf_train_list1+D_hsdf_train_list2+D_hsdf_train_list3
                                 +D_hsdf_train_list4+D_hsdf_train_list5+D_hsdf_train_list6+D_hsdf_train_list7+D_hsdf_train_list8+D_hsdf_train_list9)
    df_node_train_augmented.to_csv('processed_data\nodes_train_augmented.csv',index=False)
    df_edge_train_augmented.to_csv('processed_data\edges_train_augmented.csv',index=False)
    np.savez('processed_data\D_hsdf_train_augmented.npz', *D_hsdf_train_list_augmented)
    print("Finish data processing.")
