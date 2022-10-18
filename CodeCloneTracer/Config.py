
dirPath = "D:/projects/clone/test_project/"

url = "https://github.com/shashirajraja/onlinebookstore"

extract_from_git = False
git_first=False
# Minimum length of block to consider
minimumLengthBlock = 10

# Threshhold for considering as code clones
# Threshhold = 1 for type 2 clones
tokenSimilarityThreshold = 0.55

# Threshold for similarity measure by data flow approach
similarityDataFlowThreshold = 0.65

# Threshold for considering most frequent variables and methods
variableAndMethodsThreshold = 0.65

# Threshold while comparing dataflow of two variables and methods
dataFlowSimilaritythreshold = 0.65

# Block level can be 0 = (file level) or 1 = (method level) block_level
granularity = 'method_level'
