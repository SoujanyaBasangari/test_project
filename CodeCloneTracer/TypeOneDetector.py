def detectClone(codeBlocks):
    for codeBlockId in codeBlocks:
        codeBlock = codeBlocks[codeBlockId]
        code = codeBlock['Code']
        hashValue = 0
        singleLineCode = "".join(code)
        for ch in singleLineCode:
            if ch in (' ', '\n', '\t'):
                continue
            hashValue += ord(ch)
        codeBlock.update({"HashValue": hashValue})

    for codeBlockId in codeBlocks:
        codeBlock = codeBlocks[codeBlockId]
        hashValue = codeBlock["HashValue"]
        codeCloneIds = []
        for codeCandidateId in codeBlocks:
            if codeCandidateId == codeBlockId or codeBlocks[codeCandidateId]["HashValue"] != hashValue:
                continue
            if cloneVerification(codeBlock, codeBlocks[codeCandidateId]["Code"]):
                codeCloneIds.append(codeCandidateId)
        codeBlock.update({"CodeClones": codeCloneIds})
    return codeBlocks


def cloneVerification(Code1, Code2):
    """
    input : two list of code
    output : if two code are clones or not 
    """

    return True
