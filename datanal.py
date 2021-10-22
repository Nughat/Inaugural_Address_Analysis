import nltk
from nltk.corpus import inaugural
from nltk.corpus import stopwords
from collections import Counter
from nltk.draw.dispersion import dispersion_plot
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.text import Text
import numpy as np
import matplotlib.pyplot as plt

ids = inaugural.fileids() #the names of all the inaugural address files
def complexity(addresses):
    stopws = stopwords.words('english')
    punct = ['?',':',';',',','.','--','-',"'","!",'/','(',')',']','[']
    avgs = {}
    unqs = {}
    sent_avgs = {}
    all_long_words = {}
    for address in addresses:
        total  = 0
        long_words = 0
        uniques = set()
        #print('PRESIDENT: ',address)
        rawed = inaugural.raw(address)
        sentences = nltk.sent_tokenize(rawed)
        words = nltk.word_tokenize(rawed)
        words = [word.lower() for word in words if word not in stopws and word not in punct]
        uniques.update(words)
        for word in words:
            total += len(word)
            if len(word) > 6:
                long_words += 1
        avg = total/len(words)
        avgs[int(address[:4])] = round(avg,3)
        unq = len(uniques)/len(words)
        unqs[int(address[:4])]=round(unq,3)
        sent_avg=len(words)/len(sentences)
        sent_avgs[int(address[:4])]=round(sent_avg,3)
        all_long_words[int(address[:4])]= long_words/len(words)
    return avgs,unqs,sent_avgs,all_long_words
def syntax(addresses):
    postags = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']
    all_dic = []
    for address in addresses:
        rawed = inaugural.raw(address)
        words = nltk.word_tokenize(rawed)
        tags = nltk.pos_tag(words)
        dic = Counter(tag for word, tag in tags if tag in postags) #need to normalize based on number of words in speech
        norm_dic = Counter({k:(round((v/len(tags)),3)) for k,v in dic.items()})
        all_dic.append(norm_dic)
    return all_dic   
def lexical(addresses):
    stopws = stopwords.words('english')
    punct = ['?',':',';',',','.','--','-',"'","!",'/','(',')',']','[',"'s","''","``"]
    lex_for_all = []
    lex_for_each = []
    for address in addresses:
        rawed = inaugural.raw(address)
        words = nltk.word_tokenize(rawed)
        words = [word.lower() for word in words] #casefolding
        #stripped = rawed.translate(str.maketrans('', '', string.punctuation))
        filtered_words = [word for word in words if word not in stopws and word not in punct] #strip punctuation
        lex_for_all += filtered_words
        dic = Counter(filtered_words)
        norm_dic = Counter({k:((v/len(dic))+1) for k,v in dic.items()})
        lex_for_each.append(norm_dic) #normalized lexical counts for each address
    dic_for_all = Counter(lex_for_all)
    norm_dic_for_all = Counter({k:((round((v/len(dic_for_all)),3))+1) for k,v in dic_for_all.items()}) #normailzed counts of words across all address
    return norm_dic_for_all,lex_for_each
def dispersionCharts(targets,individual_address=None):
    if individual_address == None:
        words=inaugural.words()
        txt = Text(words)
    else:        
        rawed = inaugural.raw(individual_address)
        words = nltk.word_tokenize(rawed)
        txt = Text(words)
    plt.figure(figsize=(8, 6))
    dispersion_plot(txt, targets, ignore_case=True, title='Lexical Dispersion Plot')
    plt.show()
def meanigfulBigrams(number_of_words_between,times_appeared_together,individual_address=None):
    stopws = stopwords.words('english')
    punct = ['?',':',';',',','.','--','-',"'","!",'/','(',')',']','[',"''","``"]
    if individual_address == None:
        words=inaugural.words()
    else:     
        rawed = inaugural.raw(individual_address)
        words = nltk.word_tokenize(rawed)
    words=[word.lower() for word in words if word not in stopws and word not in punct]
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words, number_of_words_between)
    finder.apply_freq_filter(times_appeared_together)
    return finder.nbest(bigram_measures.likelihood_ratio, 10)

avgs,unqs,sent_avgs,all_long_words = complexity(ids)
all_dic = syntax(ids)
norm_dic_for_all,lex_for_each = lexical(ids)
dispersionCharts(['terror','terrorism','muslims','immigrants','aids'])
plt.savefig('foo5.png', bbox_inches='tight')

#DISPERSION 

bigrams = meanigfulBigrams(10,5,'1801-Jefferson.txt')
print(bigrams)
bigrams = meanigfulBigrams(8,2,'1861-Lincoln.txt')
print(bigrams)
bigrams = meanigfulBigrams(8,3,'1905-Roosevelt.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1933-Roosevelt.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1937-Roosevelt.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1941-Roosevelt.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1945-Roosevelt.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1953-Eisenhower.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1957-Eisenhower.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1961-Kennedy.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1965-Johnson.txt')
print(bigrams)
bigrams = meanigfulBigrams(9,2,'1981-Reagan.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1917-Wilson.txt')
print(bigrams)
bigrams = meanigfulBigrams(9,2,'1921-Harding.txt')
print(bigrams)
bigrams = meanigfulBigrams(9,2,'1925-Coolidge.txt')
print(bigrams)
bigrams = meanigfulBigrams(5,2,'1913-Wilson.txt')
print(bigrams)

#COMPLEXITY

plt.figure(figsize=(8,6))
plt.scatter(avgs.keys(), avgs.values())
plt.xticks(rotation = (45))
plt.title('Average Number of Characters Per Word In Inaugural Addresses')
plt.xlabel('Year')
plt.ylabel('Averages')
#plt.savefig('foo.png', bbox_inches='tight')
plt.figure(figsize=(8,6))
plt.scatter(unqs.keys(), unqs.values())
plt.xticks(rotation = (45))
plt.title('Unique Number of Words In Inaugural Addresses')
plt.xlabel('Year')
plt.ylabel('Frequency of Unique Words')
#plt.savefig('foo2.png', bbox_inches='tight')
plt.figure(figsize=(8,6))
plt.scatter(sent_avgs.keys(), sent_avgs.values())
plt.xticks(rotation = (45))
plt.title('Average Length of Sentences in Inaugural Addresses')
plt.xlabel('Year')
plt.ylabel('Averages')
#plt.savefig('foo3.png', bbox_inches='tight')
plt.figure(figsize=(8,6))
plt.scatter(all_long_words.keys(), all_long_words.values())
plt.xticks(rotation = (45))
plt.title('Number of "Long Words" in Inaugural Addresses')
plt.xlabel('Year')
plt.ylabel('Frequency of Long Words')
#plt.savefig('foo4.png', bbox_inches='tight')
# #SYNTAX

# a = all_dic[0]
# for i in all_dic[1:len(all_dic)]:
#     a += i
# x = [i for i in a.keys()]
# y = [round(j,3) for j in a.values()]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("POS Tags in All Inaugural Addresses")
# plt.xlabel("POS Tags")
# plt.ylabel("Frequency of Tags")
# plt.xticks(rotation=90)

# #LEXICAL

# y = [count for tag, count in norm_dic_for_all.most_common(30)]
# x = [tag for tag, count in norm_dic_for_all.most_common(30)]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("30 Most Common Words in All Inaugural Addresses")
# plt.xlabel("Words")
# plt.ylabel("Frequency of Words")
# plt.xticks(rotation=90)
# plt.ylim(1.0,1.1)

# y = [count for tag, count in lex_for_each[0].most_common(30)]
# x = [tag for tag, count in lex_for_each[0].most_common(30)]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("30 Most Common Words in Washington's 1789 Inaugural Address")
# plt.xlabel("Words")
# plt.ylabel("Frequency of Words")
# plt.xticks(rotation=90)
# plt.ylim(1.0,1.02)

# y = [count for tag, count in lex_for_each[ids.index('1865-Lincoln.txt')].most_common(30)]
# x = [tag for tag, count in lex_for_each[ids.index('1865-Lincoln.txt')].most_common(30)]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("30 Most Common Words in Lincoln's 1865 Inaugural Address")
# plt.xlabel("Words")
# plt.ylabel("Frequency of Words")
# plt.xticks(rotation=90)
# plt.ylim(1.0,1.06)

# y = [count for tag, count in lex_for_each[ids.index('1961-Kennedy.txt')].most_common(30)]
# x = [tag for tag, count in lex_for_each[ids.index('1961-Kennedy.txt')].most_common(30)]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("30 Most Common Words in Kennedy's 1961 Inaugural Address")
# plt.xlabel("Words")
# plt.ylabel("Frequency of Words")
# plt.xticks(rotation=90)
# plt.ylim(1.0,1.04)

# y = [count for tag, count in lex_for_each[ids.index('2009-Obama.txt')].most_common(30)]
# x = [tag for tag, count in lex_for_each[ids.index('2009-Obama.txt')].most_common(30)]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("30 Most Common Words in Obama's 2009 Inaugural Address")
# plt.xlabel("Words")
# plt.ylabel("Frequency of Words")
# plt.xticks(rotation=90)
# plt.ylim(1.0,1.03)

# y = [count for tag, count in lex_for_each[len(ids)-1].most_common(30)]
# x = [tag for tag, count in lex_for_each[len(ids)-1].most_common(30)]
# plt.figure(figsize=(8,6))
# plt.bar(x, y)
# plt.title("30 Most Common Words in Trump's 2017 Inaugural Address")
# plt.xlabel("Words")
# plt.ylabel("Frequency of Words")
# plt.xticks(rotation=90)
# plt.ylim(1.0,1.06)

#plt.show()



