import json
import nltk as nl
import matplotlib.pyplot as plt


with open('tag.json') as f:
    data = json.load(f)

sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nl.word_tokenize(sent)

# Creating the tokenize of sent text
tag_text = nl.pos_tag(tokens)

finalList = []

for i in tag_text:
    if i[1] in data:
        l = list((i[0], data[i[1]]))

    finalList.append(l)
# Plotting the graph
graph = nl.probability.FreqDist(tag_text)
graph.plot(30, cumulative=False)
plt.show()
    
# printing which word is what thing like (Albert is noun)
for j in finalList:
    print(j)
