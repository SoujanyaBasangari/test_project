
import embeddingModel
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
import Config

def cluster_indices(cluster_assignments):
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    return indices


def clonetracingModel(df):
    df = df.drop_duplicates(subset=['codeBlockId', 'Revision', 'codeCloneBlockId'], keep='last')
    df["unique"] = "R" + df["Revision"].astype(str) + df["codeBlockId"]
    df = df.reset_index(drop=True)
    n2v_model = embeddingModel.load_model('ccmodel')

    preprocessed_dataset = df[
        ['codeBlockId', 'codeblock_Code', 'Revision', 'codeBlock_start', 'codeBlock_end', 'codeBlock_fileinfo',
         "unique"]]

    preprocessed_dataset = preprocessed_dataset.drop_duplicates(subset=['codeBlockId', 'Revision'], keep='last')

    codeblock_Code = preprocessed_dataset['codeblock_Code'].tolist()
    # Create word embeddings
    codeblock_Code = n2v_model.vectorize_words(codeblock_Code)
    preprocessed_dataset['emdedding_codeblock_Code'] = codeblock_Code.tolist()
    data = preprocessed_dataset[['unique', 'emdedding_codeblock_Code']]
    dist = DistanceMetric.get_metric('manhattan')  # manhattan euclidean

    manhattan_distance_df = pd.DataFrame(dist.pairwise(numpy.asarray([numpy.array(xi) for xi in data['emdedding_codeblock_Code']])),columns=data.unique.unique(), index=data.unique.unique())

    # clustering
    thresh = 1.5
    clusters = hcluster.fclusterdata(manhattan_distance_df, thresh, criterion="distance")

    data['clonesets'] = clusters

    num_clusters = clusters.max()
    print("Total %d clonesets" % num_clusters)
    indices = cluster_indices(clusters)
    for k, ind in enumerate(indices):
        print("cloneset", k + 1, "is", ind)
 
    final_dataframe = pd.merge(data, df, on='unique', how='inner')

    return final_dataframe,indices


def analysis_creating_report(final_dataframe, total_files, cloning_percentage,indices,git_first):
    if git_first == False:
        output = final_dataframe[['unique', 'Revision', 'clonesets', 'codeBlockId', 'codeBlock_start', 'codeBlock_end', 'nloc',
         'codeBlock_fileinfo', 'codeCloneBlockId','change_type']]
        output = output.drop_duplicates(subset=['unique'], keep='last')
        output = output.sort_values('Revision')
        output['codeBlockId'] = output['codeBlockId'].str.replace('CodeBlock', '')
        output["codeBlockId"] = output["codeBlockId"].astype(int)
        output['Revision'] = output['Revision'].str.replace('R', '')
        output["Revision"] = output["Revision"].astype(int)
        idx = output.index
        output.sort_values(['codeBlockId', 'Revision'], inplace=True)
        output["codeBlock_start"] = pd.to_numeric(output["codeBlock_start"])#.astype(int)
        output["codeBlock_end"] =  pd.to_numeric(output["codeBlock_end"])#.astype(int)
        output["nloc"] =  pd.to_numeric(output["nloc"])
        start =  output['codeBlock_start']
        output['codeBlock_start_diffs'] = start.diff()
        end =  output['codeBlock_end']
        output['codeBlock_end_diff'] = end.diff()
        nloc =  output['nloc']
        output['nloc_diff'] = nloc.diff()
        mask = output.codeBlockId != output.codeBlockId.shift(1)
        output['codeBlock_start_diffs'][mask] = np.nan
        output['codeBlock_end_diff'][mask] = np.nan
        output['nloc_diff'][mask] = np.nan
        output.sort_values(['Revision'], ascending=True, inplace=True)
        output.reindex(idx)
        output.sort_values(["Revision", "codeBlock_fileinfo"], ascending=True).groupby("codeBlockId").first()
        output['ix'] = output.index
        ix_first = output.sort_values(["Revision", "codeBlock_fileinfo"], ascending=True).groupby("codeBlockId").first()['ix']
        output['status'] = ''
        output['status'] = output['status'].where(output['ix'].isin(ix_first), 'stable')
        output['status'] = output['status'].replace('', 'new')
        output['status'] = output['status'].replace('', 'new')
        output.loc[output.codeBlock_end_diff > 0, 'status'] = 'Modified/Added'
        output.loc[output.codeBlock_end_diff < 0, 'status'] = 'Modified/removed'
		 #output.loc[(output.status == '', 'status') |(output.change_type == 'ModificationType.ADD'), 'status']= 'new'
		 #output.loc[(output.codeBlock_end_diff > 0)|(output.change_type == 'ModificationType.MODIFY'), 'status'] = 'Modified/Added'
		 #output.loc[(output.codeBlock_end_diff < 0)|(output.change_type == 'ModificationType.MODIFY'), 'status'] = 'Modified/removed'
         # output['codeBlock_start_diffs'] = output['codeBlock_start_diffs'].replace(np.NaN, 'new')
        output['codeBlock_end_diff'] = output['codeBlock_end_diff'].replace(np.NaN, 'new')
        output['nloc_diff'] = output['nloc_diff'].replace(np.NaN, 'new')
        output = output.drop(columns=['ix'])
        output['disappearing_clone'] = 3
        output = output.set_index(["Revision", 'codeBlockId'])
        index = pd.MultiIndex.from_product(output.index.levels, names=output.index.names)
        output = output.reindex(index, fill_value=2).reset_index(level=1, drop=False).reset_index()
        output.sort_values(['codeBlockId', 'Revision'], inplace=True)
        idx = output.index
        output['disappearing_clone_diffs'] = output['disappearing_clone'].diff()
        mask = output.codeBlockId != output.codeBlockId.shift(1)
        output['disappearing_clone_diffs'][mask] = np.nan
        output['disappearing_clone_diffs'] = output['disappearing_clone_diffs'].replace(-1.0, 'disappearing_clone')
        output = output.drop(columns=['disappearing_clone'])
        output['disappearing_clone_diffs'] = output['disappearing_clone_diffs'].fillna('disappearing_clone')
        output = output.drop(output[(output['disappearing_clone_diffs'] == 0.0) & (output['status'] == 2)].index)
        output.reindex(idx)
        output = output.sort_values('Revision')
        maxvalue=output['Revision'].max()
   
    
    
    else:
        output = final_dataframe[['unique', 'Revision', 'clonesets', 'codeBlockId', 'codeBlock_start', 'codeBlock_end', 'nloc',
         'codeBlock_fileinfo', 'codeCloneBlockId','change_type','commitinfo']]
        output = output.drop_duplicates(subset=['unique'], keep='last')
        output = output.sort_values('Revision')
        output['codeBlockId'] = output['codeBlockId'].str.replace('CodeBlock', '')
        output["codeBlockId"] = output["codeBlockId"].astype(int)
        output['Revision'] = output['Revision'].str.replace('R', '')
        output["Revision"] = output["Revision"].astype(int)
        idx = output.index
        output.sort_values(['codeBlock_fileinfo', 'commitinfo'], inplace=True)
        output["codeBlock_start"] = pd.to_numeric(output["codeBlock_start"])#.astype(int)
        output["codeBlock_end"] =  pd.to_numeric(output["codeBlock_end"])#.astype(int)
        output["nloc"] =  pd.to_numeric(output["nloc"])
        start =  output['codeBlock_start']
        output['codeBlock_start_diffs'] = start.diff()
        end =  output['codeBlock_end']
        output['codeBlock_end_diff'] = end.diff()
        nloc =  output['nloc']
        output['nloc_diff'] = nloc.diff()
        mask = output.codeBlockId != output.codeBlockId.shift(1)
        output['codeBlock_start_diffs'][mask] = np.nan
        output['codeBlock_end_diff'][mask] = np.nan
        output['nloc_diff'][mask] = np.nan
        output.sort_values(['commitinfo'], ascending=True, inplace=True)
        output.reindex(idx)
        output.sort_values(["commitinfo", "codeBlock_fileinfo"], ascending=True).groupby("codeBlockId").first()
        output['ix'] = output.index
        ix_first = output.sort_values(["commitinfo", "codeBlock_fileinfo"], ascending=True).groupby("codeBlockId").first()['ix']
        output['status'] = ''
        output['status'] = output['status'].where(output['ix'].isin(ix_first), 'stable')
        output['status'] = output['status'].replace('', 'new')
        output['status'] = output['status'].replace('', 'new')
        output.loc[output.codeBlock_end_diff > 0, 'status'] = 'Modified/Added'
        output.loc[output.codeBlock_end_diff < 0, 'status'] = 'Modified/removed'
		 #output.loc[(output.status == '', 'status') |(output.change_type == 'ModificationType.ADD'), 'status']= 'new'
		 #output.loc[(output.codeBlock_end_diff > 0)|(output.change_type == 'ModificationType.MODIFY'), 'status'] = 'Modified/Added'
		 #output.loc[(output.codeBlock_end_diff < 0)|(output.change_type == 'ModificationType.MODIFY'), 'status'] = 'Modified/removed'
         # output['codeBlock_start_diffs'] = output['codeBlock_start_diffs'].replace(np.NaN, 'new')
        output['codeBlock_end_diff'] = output['codeBlock_end_diff'].replace(np.NaN, 'new')
        output['nloc_diff'] = output['nloc_diff'].replace(np.NaN, 'new')
        output = output.drop(columns=['ix'])
        output['disappearing_clone'] = 3
        output = output.set_index(["commitinfo", 'codeBlockId'])
        index = pd.MultiIndex.from_product(output.index.levels, names=output.index.names)
        output = output.reindex(index, fill_value=2).reset_index(level=1, drop=False).reset_index()
        output.sort_values(['codeBlockId', 'commitinfo'], inplace=True)
        idx = output.index
        output['disappearing_clone_diffs'] = output['disappearing_clone'].diff()
        mask = output.codeBlockId != output.codeBlockId.shift(1)
        output['disappearing_clone_diffs'][mask] = np.nan
        output['disappearing_clone_diffs'] = output['disappearing_clone_diffs'].replace(-1.0, 'disappearing_clone')
        output = output.drop(columns=['disappearing_clone'])
        output['disappearing_clone_diffs'] = output['disappearing_clone_diffs'].fillna('disappearing_clone')
        output = output.drop(output[(output['disappearing_clone_diffs'] == 0.0) & (output['status'] == 2)].index)
        output.reindex(idx)
        output = output.sort_values('commitinfo')
        maxvalue=output['commitinfo'].nunique()#.max()
    
    
    
    granularity = Config.granularity
    path = str(Config.dirPath)+str(granularity)+'.txt'
    
    with open(path, 'w') as f:
        for k, ind in enumerate(indices):
            f.write("cloneset{}\n".format(k + 1))
            f.write("{}\n".format(output.iloc[ind]['unique'].to_list()))
            f.write("\n") 
  
        f.write("cloning_percentage = {}\n".format(cloning_percentage))

        f.write("FILE LEVEL INFORMATION")
        f.write("total_files = {}\n".format(total_files))
        
        maxvalue=output['Revision'].max()
        final_revision = output[output.Revision == 4]
        f.write("final_revision = {}\n".format(maxvalue))

        maxvalue=output['Revision'].max()
        final_revision = output[output.Revision == maxvalue]

        files_containing_clones = len(pd.unique(final_revision['codeBlock_fileinfo']))#final_revision.codeBlock_fileinfo.count()
        f.write("files_containing_clones = {}\n".format(files_containing_clones))

        added_files = len(pd.unique(final_revision[final_revision['status']== 'new']['codeBlock_fileinfo']))#
        f.write("added_files = {}\n".format(added_files))

        deleted_files_df = final_revision[(final_revision.disappearing_clone_diffs == 'disappearing_clone') & (final_revision.status == 2)]

        deleted_files =  len(pd.unique(deleted_files_df.codeBlock_fileinfo))
        f.write("deleted_files = {}\n".format(deleted_files))
        

        f.write("CLONESETS INFORMATION")

        total_clone_sets = len(pd.unique(final_revision.clonesets))
        f.write("total_clone_sets = {}\n".format(total_clone_sets))

        stable_clonesets = len(pd.unique(final_revision[final_revision['status']== 'stable']['clonesets']))
        f.write("stable_clonesets = {}\n".format(stable_clonesets))

        new_clonesets = len(pd.unique(final_revision[final_revision['status']== 'new']['clonesets']))
        f.write("new_clonesets = {}\n".format(new_clonesets))

        deleted_clonesets = len(pd.unique(final_revision[(final_revision.disappearing_clone_diffs == 'disappearing_clone') & (final_revision.status == 2)]['clonesets']))
        f.write("deleted_clonesets = {}\n".format(deleted_clonesets))

        final_revision['status']=final_revision['status'].astype(str)

        changed_clonesets = len(pd.unique(final_revision[final_revision['status'].str.contains('Modified')]['clonesets']))

        f.write("changed_clonesets = {}\n".format(changed_clonesets))

        f.write("CODECLONES INFORMATION")

        total_codeclones= len(pd.unique(final_revision.codeBlockId))

        f.write("total_codeclones = {}\n".format(total_codeclones))
        stable_codeclones = len(pd.unique(final_revision[final_revision['status']== 'stable']['codeBlockId']))
        f.write("stable_codeclones = {}\n".format(stable_codeclones))

        new_codeclones = len(pd.unique(final_revision[final_revision['status']== 'new']['codeBlockId']))
        f.write("new_codeclones = {}\n".format(new_codeclones))

        deleted_codeclones = len(pd.unique(final_revision[(final_revision.disappearing_clone_diffs == 'disappearing_clone') & (final_revision.status == 2)]['codeBlockId']))
        f.write("deleted_codeclones = {}\n".format(deleted_codeclones))

        changed_codeclones = len(pd.unique(final_revision[final_revision['status'].str.contains('Modified')]['codeBlockId']))
        f.write("changed_codeclones = {}\n".format(changed_codeclones))

        f.close()

    return output
