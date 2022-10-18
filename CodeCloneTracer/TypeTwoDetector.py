import re
import Mapping


def detectClone(codeBlocks):
    for codeBlockId in codeBlocks:
        codeBlock = codeBlocks[codeBlockId]
        code = codeBlock['Code']
        list_tokens = []
        for line in code:
            line = re.sub(r"(\".*?\"|\'.*?\')", "STRING_LITERAL", line)
            line = re.sub(r"[+-]?((\d+(\.\d+)?)|(\.\d+))", "INTEGER_LITERAL", line)
            list_line = re.split('(\W)', line)
            # Now considering : ; as well
            # lst_line = re.findall(r"[\w']+", line)
            # Now will remove delimiters

            for unit in list_line:
                unit = unit.strip()
                if unit in Mapping.delimiters:
                    continue
                elif unit in Mapping.mapping.keys():
                    list_tokens.append(Mapping.mapping[unit])
                else:
                    list_tokens.append("TOKEN")

        dict_tokens = {}
        for token in list_tokens:
            if token in dict_tokens.keys():
                dict_tokens[token] = dict_tokens[token] + 1
            else:
                dict_tokens[token] = 1
        codeBlock.update({"Tokens": dict_tokens})

    for codeBlockId in codeBlocks:
        codeBlock = codeBlocks[codeBlockId]
        tokens = codeBlock["Tokens"]
        codeCloneIds = []
        for codeCandidateId in codeBlocks:
            if codeCandidateId == codeBlockId:
                continue
            if cloneVerification(tokens, codeBlocks[codeCandidateId]["Tokens"]):
                codeCloneIds.append(codeCandidateId)
        codeBlock.update({"CodeClones": codeCloneIds})
    return codeBlocks


def cloneVerification(Tokens1, Tokens2):
    """
    input : two list of code
    output : if two code are clones or not 
    """
    if len(Tokens1.keys()) != len(Tokens2.keys()):
        return False

    for key in Tokens1.keys():
        if (key not in Tokens2.keys() or Tokens1[key] != Tokens2[key]) and key != "TOKEN":
            return False
    return True
