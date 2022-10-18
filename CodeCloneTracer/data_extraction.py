from datetime import datetime
import itertools
import logging
import os
# import GetFunctions
import re
import sys
import traceback
import CloneDetector
import javalang
import datetime
import Config
import pandas as pd
from pydriller import Repository
global found_parent
from pydriller.metrics.process.lines_count import LinesCount

def extractMethods(url):
    allFilesMethodsBlocks = {}
    linesofcode = 0
    blocksSoFar = 0
    commits = []
    latest_commit = ''
    first_commit = ''
    for commit in Repository(url).traverse_commits():
      commits.append(commit.hash)
      latest_commit = commits[-1]
      first_commit = commits[0]

    metric = LinesCount(path_to_repo=url,from_commit=first_commit,to_commit=latest_commit)
    total_lines = metric.count()
    print('Total lines : {}'.format(sum(total_lines.values())))
    total_lines = sum(total_lines.values())
    codeBlocks={}
    for commit in Repository(url,only_commits=[latest_commit]).traverse_commits():#,only_commits=[latest_commit]
        filename_list=[]
        for i in commit.modified_files:
            if i.filename.endswith('java'): 
                if Config.granularity == "method_level":   
                    filename_list.append(i.filename)
                    originalcode = str(i.source_code).replace('\r', '').replace('\t', '').split('\n')
                    linesofcode = linesofcode + len(originalcode)
                    codeBlocks = methodLevelBlocks(originalcode) 
                elif Config.granularity == 'file_level':
                    filename_list.append(i.filename)
                    originalcode = str(i.source_code).replace('\r', '').replace('\t', '').split('\n')
                    linesofcode = linesofcode + len(originalcode)
                    codeBlocks = fileLevelBlocks(originalcode)
                else:
                    filename_list.append(i.filename)
                    originalcode = str(i.source_code).replace('\r', '').replace('\t', '').split('\n')
                    linesofcode = linesofcode + len(originalcode)
                    codeBlocks =  normalized_codeblocks(originalcode)
                if len(codeBlocks) == 0:
                    continue
                for codeBlock in codeBlocks:
                    if len(codeBlock) == 0:
                        continue
                    codeBlock.update({"FileInfo": i.filename})
                    codeBlock.update({"change_type": i.change_type})
                    codeBlock.update({"old_path": i.old_path})
                    codeBlock.update({"new_path": i.new_path})
                    codeBlock.update({"source_code": i.source_code})
                    codeBlock.update({"committer_date":commit.committer_date})
                    codeBlock.update({"nloc": i.nloc})
                    codeBlock.update({"commitinfo": commit.hash})
                    blocksSoFar += 1
                    allFilesMethodsBlocks["CodeBlock" + str(blocksSoFar)] = codeBlock 
    previous_clones = pd.DataFrame(
        columns=['codeBlockId', 'codeBlock_start', 'codeBlock_end', 'codeBlock_fileinfo', 'codeblock_Code','tokens',
                 'codeCloneBlockId',
                 'codeCloneBlock_Fileinfo', 'Similarity_Tokens', 'Similarity_Variable_Flow',
                 'Similarity_MethodCall_Flow', 'commitinfo', 'nloc', 'Revision','change_type','committer_date'])
    
    previous_file_name = Config.granularity + 'gittracking.csv'
   
    if os.path.isfile(previous_file_name): 
      previous_dataset = pd.read_csv(previous_file_name, index_col=0)
      for index, row in previous_dataset.iterrows():
        for codeBlock in codeBlocks:
          codeBlock.update({"Code": row['codeblock_Code']})
          codeBlock.update({"Start":row['codeBlock_start']})
          codeBlock.update({"End": row['codeBlock_end']})
          codeBlock.update({"FileInfo": row['codeBlock_fileinfo']})
          codeBlock.update({"committer_date":row['committer_date']})
          codeBlock.update({"nloc": row['nloc']})
          allFilesMethodsBlocks["CodeBlock" + str("old"+row['codeBlockId'])] = codeBlock 
      
    print("total code blocks",len(allFilesMethodsBlocks),linesofcode)
    cloneBlocks, codeclonelines = CloneDetector.detectClone(allFilesMethodsBlocks)
 
    current_dataset = dataset_creation(cloneBlocks)
    current_dataset['nloc'] = current_dataset['nloc'].astype(int)
    #codeclonelines = current_dataset.groupby('codeBlockId').apply(lambda x: x['nloc'].unique()).sum()#['nloc']
    current_dataset['sum'] = current_dataset.groupby('codeBlockId')['nloc'].sum()
    print(codeclonelines,type(codeclonelines),codeclonelines)
    codeclonelines = current_dataset['sum'].sum()
    print("detecting code clones",len(cloneBlocks),codeclonelines)
    current_dataset = current_dataset[current_dataset["codeBlockId"].str.contains("old") == False]
    print("Transforming detected code blocks into dataset",current_dataset.shape)
   
    if os.path.isfile(previous_file_name): 
        previous_dataset = pd.read_csv(previous_file_name, index_col=0)
        revision = previous_dataset.Revision.unique()
        
        previous_clones = previous_dataset[~previous_dataset.codeBlock_fileinfo.isin(current_dataset.codeBlock_fileinfo)]
        frames = [current_dataset,previous_clones]
        print("Revision", revision,revision[0] + 1)
        current_dataset = pd.concat([current_dataset, previous_dataset])
        current_dataset['Revision'] = revision[0] + 1
        current_dataset = current_dataset.loc[current_dataset.astype(str).drop_duplicates().index]

    else:
        print("First version, no cloning result exists")
        print("Revision", 1)
        current_dataset['Revision'] = 1
  
    current_dataset = current_dataset.convert_dtypes()
    all_columns = list(current_dataset)  # Creates list of all column headers
    current_dataset[all_columns] = current_dataset[all_columns].astype(str)
    current_dataset = current_dataset.loc[current_dataset.astype(str).drop_duplicates().index]
    current_dataset['datetime'] = datetime.datetime.now()
    current_dataset = current_dataset.reset_index(drop=True)
    current_dataset = current_dataset.drop_duplicates()
    current_dataset = current_dataset.reset_index(drop=True)
    current_dataset.to_csv(Config.granularity + 'gittracking.csv')
    return current_dataset, total_lines, codeclonelines,len(filename_list)

def extractMethods_first(url):
    allFilesMethodsBlocks = {}
    linesofcode = 0
    blocksSoFar = 0
    commits = []
    latest_commit = ''
    first_commit = ''
    for commit in Repository(url).traverse_commits():
      commits.append(commit.hash)
      latest_commit = commits[-1]
      first_commit = commits[0]

    metric = LinesCount(path_to_repo=url,from_commit=first_commit,to_commit=latest_commit)
    total_lines = metric.count()
    print('Total lines : {}'.format(sum(total_lines.values())))
    total_lines = sum(total_lines.values())
    codeBlocks={}
    for commit in Repository(url,only_commits=[latest_commit]).traverse_commits():#,only_commits=[latest_commit]
        filename_list=[]
        for i in commit.modified_files:
            if i.filename.endswith('java'): 
                if Config.granularity == "method_level":   
                    filename_list.append(i.filename)
                    originalcode = str(i.source_code).replace('\r', '').replace('\t', '').split('\n')
                    linesofcode = linesofcode + len(originalcode)
                    codeBlocks = methodLevelBlocks(originalcode) 
                elif Config.granularity == 'file_level':
                    filename_list.append(i.filename)
                    originalcode = str(i.source_code).replace('\r', '').replace('\t', '').split('\n')
                    linesofcode = linesofcode + len(originalcode)
                    codeBlocks = fileLevelBlocks(originalcode)
                else:
                    filename_list.append(i.filename)
                    originalcode = str(i.source_code).replace('\r', '').replace('\t', '').split('\n')
                    linesofcode = linesofcode + len(originalcode)
                    codeBlocks =  normalized_codeblocks(originalcode)
                if len(codeBlocks) == 0:
                    continue
                for codeBlock in codeBlocks:
                    if len(codeBlock) == 0:
                        continue
                    codeBlock.update({"FileInfo": i.filename})
                    codeBlock.update({"change_type": i.change_type})
                    codeBlock.update({"old_path": i.old_path})
                    codeBlock.update({"new_path": i.new_path})
                    codeBlock.update({"source_code": i.source_code})
                    codeBlock.update({"committer_date":commit.committer_date})
                    codeBlock.update({"nloc": i.nloc})
                    codeBlock.update({"commitinfo": commit.hash})
                    blocksSoFar += 1
                    allFilesMethodsBlocks["CodeBlock" + str(blocksSoFar)] = codeBlock 
    previous_clones = pd.DataFrame(
        columns=['codeBlockId', 'codeBlock_start', 'codeBlock_end', 'codeBlock_fileinfo', 'codeblock_Code','tokens',
                 'codeCloneBlockId',
                 'codeCloneBlock_Fileinfo', 'Similarity_Tokens', 'Similarity_Variable_Flow',
                 'Similarity_MethodCall_Flow', 'commitinfo', 'nloc', 'Revision','change_type','committer_date'])
    
    previous_file_name = Config.granularity + 'analysistracking.csv'
   
    if os.path.isfile(previous_file_name): 
      previous_dataset = pd.read_csv(previous_file_name, index_col=0)
      for index, row in previous_dataset.iterrows():
        for codeBlock in codeBlocks:
          codeBlock.update({"Code": row['codeblock_Code']})
          codeBlock.update({"Start":row['codeBlock_start']})
          codeBlock.update({"End": row['codeBlock_end']})
          codeBlock.update({"FileInfo": row['codeBlock_fileinfo']})
          codeBlock.update({"committer_date":row['committer_date']})
          codeBlock.update({"nloc": row['nloc']})
          allFilesMethodsBlocks["CodeBlock" + str("old"+row['codeBlockId'])] = codeBlock 
      
    print("total code blocks",len(allFilesMethodsBlocks),linesofcode)
    cloneBlocks, codeclonelines = CloneDetector.detectClone(allFilesMethodsBlocks)
 
    current_dataset = dataset_creation(cloneBlocks)
    current_dataset['nloc'] = current_dataset['nloc'].astype(int)
    #codeclonelines = current_dataset.groupby('codeBlockId').apply(lambda x: x['nloc'].unique()).sum()#['nloc']
    codeclonelines = current_dataset.groupby('codeBlockId')['nloc'].sum()
    codeclonelines = codeclonelines['nloc'].sum()
    print("detecting code clones",len(cloneBlocks),codeclonelines)
    
    current_dataset = current_dataset[current_dataset["codeBlockId"].str.contains("old") == False]
    print("Transforming detected code blocks into dataset",current_dataset.shape)

    if os.path.isfile(previous_file_name): 
        previous_dataset = pd.read_csv(previous_file_name, index_col=0)
        revision = previous_dataset.Revision.unique()
        
        previous_clones = previous_dataset[~previous_dataset.codeBlock_fileinfo.isin(current_dataset.codeBlock_fileinfo)]
        frames = [current_dataset,previous_clones]
        print("Revision", revision,revision[0] + 1)
        current_dataset = pd.concat([current_dataset, previous_dataset])
        current_dataset['Revision'] = revision[0] + 1
        current_dataset = current_dataset.loc[current_dataset.astype(str).drop_duplicates().index]

    else:
        print("First version, no cloning result exists")
        print("Revision", 1)
        current_dataset['Revision'] = 1
  
    current_dataset = current_dataset.convert_dtypes()
    all_columns = list(current_dataset)  # Creates list of all column headers
    current_dataset[all_columns] = current_dataset[all_columns].astype(str)
    current_dataset = current_dataset.loc[current_dataset.astype(str).drop_duplicates().index]
    current_dataset['datetime'] = datetime.datetime.now()
    current_dataset = current_dataset.reset_index(drop=True)
    current_dataset = current_dataset.drop_duplicates()
    current_dataset = current_dataset.reset_index(drop=True)
    current_dataset.to_csv(Config.granularity + 'analysistracking.csv')
    return current_dataset,total_lines, codeclonelines,len(filename_list)


def extractMethodsAllFiles(listOfFiles):
    allFilesMethodsBlocks = {}
    blocksSoFar = 0
    linesofcode = 0
    codeBlocks = {}
    print("data extraction from source code")
    for filePath in listOfFiles:
        file = open(filePath, 'r',encoding = "ISO-8859-1")# encoding='utf-8' encoding = "ISO-8859-1"
        originalCode = file.readlines()
        file.close()
        if Config.granularity == 'method_level':

            linesofcode = linesofcode + len(originalCode)
            codeBlocks = methodLevelBlocks(originalCode)

        elif Config.granularity == 'file_level':
            linesofcode = linesofcode + len(originalCode)
            codeBlocks = fileLevelBlocks(originalCode)
        else:
            linesofcode = linesofcode + len(originalCode)
            codeBlocks =  normalized_codeblocks(originalCode)
        if len(codeBlocks) == 0:
            continue
        for codeBlock in codeBlocks:
            if len(codeBlock) == 0:
                continue
            codeBlock.update({"FileInfo": filePath})
            codeBlock.update({"nloc": len(codeBlock)})
            codeBlock.update({"source_code": originalCode})
            codeBlock.update({"change_type": 'NA'})
            codeBlock.update({"commitinfo":  'NA'})
            codeBlock.update({"committer_date":  datetime.date})
            blocksSoFar += 1
            allFilesMethodsBlocks["CodeBlock" + str(blocksSoFar)] = codeBlock
    
  
    granularity = Config.granularity
    print("total code blocks",len(allFilesMethodsBlocks),linesofcode)
    cloneBlocks, codeclonelines = CloneDetector.detectClone(allFilesMethodsBlocks)
    print("detecting code clones",len(cloneBlocks),codeclonelines)
    previous_file_name = str(Config.dirPath)+ granularity + 'tracking.csv'
    current_dataset = dataset_creation(cloneBlocks)
    print("Transforming detected code blocks into dataset",current_dataset.shape)
    previous_dataset = pd.DataFrame()
    previous_clones = pd.DataFrame(
        columns=['codeBlockId', 'codeBlock_start', 'codeBlock_end', 'codeBlock_fileinfo', 'codeblock_Code','tokens',
                 'codeCloneBlockId',
                 'codeCloneBlock_Fileinfo', 'Similarity_Tokens', 'Similarity_Variable_Flow',
                 'Similarity_MethodCall_Flow', 'commitinfo', 'nloc', 'Revision','change_type','committer_date'])
    
    if os.path.isfile(previous_file_name):  # previous_file_name.exists():
        previous_dataset = pd.read_csv(previous_file_name, index_col=0)
        revision = previous_dataset.Revision.unique()
        print("Revision", revision,revision[0] )
        previous_clones = previous_dataset[~previous_dataset.codeBlock_fileinfo.isin(current_dataset.codeBlock_fileinfo)]
        frames = [current_dataset,previous_clones]
        current_dataset['Revision'] = revision[0] + 1
        current_dataset = pd.concat([current_dataset, previous_dataset])
        current_dataset = current_dataset.loc[current_dataset.astype(str).drop_duplicates().index]

    else:
        print("First version, no cloning result exists")
        print("Revision", 1)
        current_dataset['Revision'] = 1

    current_dataset = current_dataset.convert_dtypes()
    all_columns = list(current_dataset)  
    current_dataset[all_columns] = current_dataset[all_columns].astype(str)
    current_dataset = current_dataset.loc[current_dataset.astype(str).drop_duplicates().index]
    current_dataset['datetime'] = datetime.datetime.now()
    current_dataset = current_dataset.reset_index(drop=True)
    current_dataset = current_dataset.drop_duplicates()
    current_dataset = current_dataset.reset_index(drop=True)
    current_dataset.to_csv(Config.dirPath+ granularity + 'tracking.csv')

    return current_dataset, linesofcode, codeclonelines


def dataset_creation(codeBlocks):
    df = pd.DataFrame(
        columns=['codeBlockId', 'codeBlock_start', 'codeBlock_end', 'codeBlock_fileinfo', 'codeblock_Code','tokens',
                 'codeCloneBlockId',
                 'codeCloneBlock_Fileinfo', 'Similarity_Tokens', 'Similarity_Variable_Flow',
                 'Similarity_MethodCall_Flow', 'nloc','change_type','commitinfo','committer_date'])

    output = []
    for codeBlockId in codeBlocks:
        codeBlock = codeBlocks[codeBlockId]
        for codeCloneBlockData in codeBlock["CodeClones"]:
            codeCloneBlockId = codeCloneBlockData["codeCandidateId"]
            codeCloneBlock = codeBlocks[codeCloneBlockId]
            codeCloneSimilarity = codeCloneBlockData["Similarity"]
            output.append(
                [codeBlockId, str(codeBlock["Start"]), str(codeBlock["End"]), codeBlock["FileInfo"], codeBlock["Code"],
                 codeBlock["Tokens"],
                 codeCloneBlockData["codeCandidateId"], codeCloneBlock["FileInfo"], str(codeCloneSimilarity[0]),
                 str(codeCloneSimilarity[1]), str(codeCloneSimilarity[2]), str(codeBlock["nloc"]), str(codeBlock["change_type"]),str(codeBlock["commitinfo"]),
                 str(codeBlock["committer_date"])
                 ])
    for index, x in enumerate(output):
        a_row = pd.Series([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10],x[11],x[12],x[13],x[14]],
                          index=['codeBlockId', 'codeBlock_start', 'codeBlock_end', 'codeBlock_fileinfo',
                                 'codeblock_Code', 'tokens','codeCloneBlockId',
                                 'codeCloneBlock_Fileinfo', 'Similarity_Tokens', 'Similarity_Variable_Flow',
                                 'Similarity_MethodCall_Flow', 'nloc','change_type','commitinfo','committer_date'])
        row_df = pd.DataFrame([a_row])
        df = df.append(row_df)

    return df
def normalized_codeblocks(originalCode):
    """
    input : originalCode
    output : blocks using file level
    """
    allCodeBlocks = []
    commentsRemovedCode = removeCommentsFromCode(originalCode)
    
    blocks = [l.split(',') for l in ','.join(commentsRemovedCode).split('}')]
    
    startLine = 1
    endLine = 1
    for index,i in enumerate(blocks):  
      flat_list = [subblocks.replace(' ', '') for subblocks in blocks[index]]
      
      flat_list = list(filter(None, flat_list))
      endLine = endLine +len(flat_list)
      if len(flat_list) > 0:
          allCodeBlocks.append({"Start": startLine, "End": endLine, "Code": flat_list})
      startLine = startLine+ len(flat_list)
    
    return allCodeBlocks

def fileLevelBlocks(originalCode):
    """
    input : originalCode
    output : blocks using file level
    """

    allCodeBlocks = []
    commentsRemovedCode = removeCommentsFromCode(originalCode)
    startLine = 1
    endLine = len(commentsRemovedCode)
    allCodeBlocks.append(
        {"Start": startLine, "End": endLine, "Code": commentsRemovedCode})
    return allCodeBlocks


def methodLevelBlocks(originalCode):
    """
    input : originalCode
    output : blocks using method level
    """
    commentsRemovedCode = removeCommentsFromCode(originalCode)
    codeInSingleLine = "\n".join(commentsRemovedCode)

    output = method_extractor(codeInSingleLine)

    allCodeBlocks = []
    if output[0] == None:
        return allCodeBlocks
    for i in range(len(output[0])):
        if abs(output[0][i][1] - output[0][i][0]) < Config.minimumLengthBlock - 1:
            continue
        allCodeBlocks.append(
            {"Start": output[0][i][0], "End": output[0][i][1], "Code": output[1][i].split('\n')})

    return allCodeBlocks


# get all lines of code before detection
# get all clone code lines
# send code blocks to dataset creation

def removeCommentsFromCode(originalCode):
    """
    input : original Code
    output : code without comments 
    """

    DEFAULT = 1
    ESCAPE = 2
    STRING = 3
    ONE_LINE_COMMENT = 4
    MULTI_LINE_COMMENT = 5

    mode = DEFAULT
    strippedCode = []
    for line in originalCode:
        strippedLine = ""
        idx = 0
        while idx < len(line):
            subString = line[idx: min(idx + 2, len(line))]
            c = line[idx]
            if mode == DEFAULT:
                mode = MULTI_LINE_COMMENT if subString == "/*" else ONE_LINE_COMMENT if subString == "//" else STRING if c == '\"' else DEFAULT
            elif mode == STRING:
                mode = DEFAULT if c == '\"' else ESCAPE if c == '\\' else STRING
            elif mode == ESCAPE:
                mode = STRING
            elif mode == ONE_LINE_COMMENT:
                mode = DEFAULT if c == '\n' else ONE_LINE_COMMENT
                idx += 1
                continue
            elif mode == MULTI_LINE_COMMENT:
                mode = DEFAULT if subString == "*/" else MULTI_LINE_COMMENT
                idx += 2 if mode == DEFAULT else 1
                continue
            strippedLine += c if mode < 4 else ""
            idx += 1
        if len(strippedLine) > 0 and strippedLine[-1] == '\n':
            strippedLine = strippedLine[:-1]
        # strippedLine = re.sub('\t| +', ' ', strippedLine)
        strippedCode.append(strippedLine)
    return strippedCode


try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

re_string = re.escape("\"") + '.*?' + re.escape("\"")


def getFunctions(filestring, comment_inline_pattern=".*?$"):
    method_string = []
    method_pos = []
    method_name = []

    global found_parent
    found_parent = []

    tree = None

    try:
        tree = javalang.parse.parse(filestring)
        package = tree.package
        if package is None:
            package = 'DefaultPackage'
        else:
            package = package.name
    except Exception as e:
        # logging.warning('Traceback:' + traceback.print_exc())
        return (None, None, [])

    file_string_split = filestring.split('\n')
    nodes = itertools.chain(tree.filter(
        javalang.tree.ConstructorDeclaration), tree.filter(javalang.tree.MethodDeclaration))

    for path, node in nodes:
        name = '.' + node.name
        for i, var in enumerate(reversed(path)):
            if isinstance(var, javalang.tree.ClassDeclaration):
                if len(path) - 3 == i:  # Top most
                    name = '.' + var.name + check_repetition(var, var.name) + name
                else:
                    name = '$' + var.name + check_repetition(var, var.name) + name
            if isinstance(var, javalang.tree.ClassCreator):
                name = '$' + var.type.name + \
                       check_repetition(var, var.type.name) + name
            if isinstance(var, javalang.tree.InterfaceDeclaration):
                name = '$' + var.name + check_repetition(var, var.name) + name
       
        args = []
        for t in node.parameters:
            dims = []
            if len(t.type.dimensions) > 0:
                for e in t.type.dimensions:
                    dims.append("[]")
            dims = "".join(dims)
            args.append(t.type.name + dims)
        args = ",".join(args)

        fqn = ("%s%s(%s)") % (package, name, args)
       

        (init_line, b) = node.position
        method_body = []
        closed = 0
        openned = 0


        for line in file_string_split[init_line - 1:]:
        
            line_re = re.sub(comment_inline_pattern, '',
                             line, flags=re.MULTILINE)
            line_re = re.sub(re_string, '', line_re, flags=re.DOTALL)

            closed += line_re.count('}')
            openned += line_re.count('{')
            if (closed - openned) == 0 and openned > 0:
                method_body.append(line)
                break
            else:
                method_body.append(line)

     

        end_line = init_line + len(method_body) - 1
        method_body = '\n'.join(method_body)

        method_pos.append((init_line, end_line))
        method_string.append(method_body)

        method_name.append(fqn)

    if (len(method_pos) != len(method_string)):
        return (None, None, method_name)
    else:
        return (method_pos, method_string, method_name)


def check_repetition(node, name):
    before = -1
    i = 0
    for (obj, n, value) in found_parent:
        if obj is node:
            if value == -1:
                return ''
            else:
                return '_' + str(value)
        else:
            i += 1
        if n == name:
            before += 1
    found_parent.append((node, name, before))
    if before == -1:
        return ''
    else:
        return '_' + str(before)


def method_extractor(file):
    methodsInfo = []

    FORMAT = '[%(levelname)s] (%(threadName)s) %(message)s'

    config = ConfigParser()


    separators = "; . [ ] ( ) ~ ! - + & * / % < > ^ | ? { } = # , \" \\ : $ ' ` @"
    comment_inline = "#"
    comment_inline_pattern = comment_inline + '.*?$'

    return getFunctions(file, comment_inline_pattern)


def getAllFilesUsingFolderPath(folderPath):
    allFilesInFolder = []
    fileCount = 0

    for subdir, dirs, files in os.walk(folderPath):
        for fileName in files:
            fileCount += 1
            if fileName.split(".")[-1] != "java":
                continue
            fileFullPath = os.path.join(subdir, fileName)
            allFilesInFolder.append(fileFullPath)
          
    return allFilesInFolder
