keywordsList = """abstract continue for new switch assert default goto 
package synchronized boolean do if private this break double implements 
protected throw byte import public throws case enum	instanceof return 
transient catch extends int short try char final interface static void
class finally long strictfp volatile const float native super while String
STRING_LITERAL INTEGER_LITERAL
""".split()
mapping = {keywordsList[i]: "TOKEN" + str(i) for i in range(0, len(keywordsList))}
keywords = {keywordsList[i]: keywordsList[i]
            for i in range(0, len(keywordsList))}
symbols = ["", "+", "-", "*", "/", " ", "{", "}", ";", ":", ".",
           "\t", "\n", ",", "(", ")", "[", "]", "=", ">", "<", " ", "!", "\\", "|", "&", "%", "^", "~", "`", "?"]
delimiters = ["+", "-", "*", "/", "{", "}", ";", "\t",
              ":", ",", "(", ")", "[", "]", "=", ">", "<", "!", "\\", "|", "&", "%", '^', "~", "`", "?"]
