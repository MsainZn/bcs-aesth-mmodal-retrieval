# import PreprocessingUtilities as pss
# import pandas as pd
# import numpy as np

# #Required Paths
# current_directory = os.getcwd()
# images_path='../data/images/'
# csvs_path ='../data/csvs/'
# pickle_path = current_directory + '/../data/pickles/'
# path_save = '../bin/'
# favorite_image_info = csvs_path + 'favorite_image_info.csv'
# patient_info = csvs_path + 'patient_info.csv'
# patient_images_info = csvs_path + 'patient_images.csv'
# catalogue_info = csvs_path + 'catalogue_info.csv'
# catalogue_user_info = csvs_path + 'catalogue_user_info.csv'



# def get_tabular_features_filtered(patient_info_csv, id, selected_features= ['Patient Height', 'Patient Weight', 'Patient Birthday', 'Bra Size', 'Bra Cup']):

#     full_info = pss.get_tabular_data_from_id(patient_info_csv, id)
#     selected_info = full_info[selected_features]
    
#     if selected_info.isna().any().any():
#         return None

#     if 'Patient Birthday' in selected_features:
#         selected_info.loc[:, 'Patient Birthday']  = selected_info['Patient Birthday'].apply(pss.convert_birthday_to_age)

#     if 'Bra Size' in selected_features:
#         selected_info.loc[:, 'Bra Size']  = selected_info['Bra Size'].apply(pss.convert_brasize_to_underbust)

#     if 'Bra Cup' in selected_features:
#         selected_info.loc[:, 'Bra Cup']  = selected_info['Bra Cup'].apply(pss.convert_bracup_to_oubustdiff)
    
#     if selected_info.isna().any().any():
#         return None
    
#     return selected_info

# queries_id, neighbours_id = pss.get_catalogues_ids_from_csv(catalogue_info, -1)

# for idx, q in enumerate(queries_id):
#     for jdx, n in enumerate(neighbours_id[idx]):
#         if q == n:
#             neighbours_id[idx] = np.delete(neighbours_id[idx], jdx) #neighbours_id[idx].remove(n)

# for idx in range(len(queries_id)): 
#     data = get_tabular_features_filtered(patient_info, id)
#     if 