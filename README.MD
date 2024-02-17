# Epam project


# DS part
##  Conclusions from EDA
### Describing the data

...Data columns (total 2 columns):
     Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   review     40000 non-null  object
 1   sentiment  40000 non-null  object

review sentiment
count                                               40000     40000
unique                                              39728         2
top     Loved today's show!!! It was a variety and not...  positive
000
count        40000
unique           2
top       positive
freq         20000...


- The sentiment distribution is balanced with 20,000 positive and likely 20,000 negative reviews. This balance is advantageous for training a sentiment analysis model.

- No missing values are observed in either the "review" or "sentiment" columns, indicating good data quality.

### Distribution of data

...count    40000.000000
mean      1310.549450
std        987.955229
min         41.000000
25%        699.000000
50%        971.000000
75%       1595.000000
max      13704.000000...

- The distribution of the data appears to be right-skewed, as the mean (1310.55) is greater than the median (971), and the maximum value (13704) is considerably higher than the mean.

- The standard deviation (987.96) indicates moderate variability in the data.

- The interquartile range (IQR), the range between the 25th and 75th percentiles (899 to 1595), provides insights into the spread of the middle 50% of the data.

- There might be outliers present in the dataset, given the large difference between the maximum value and the 75th percentile.

### Top words and their frequencies

...Top words and their frequencies(for positive reviews):
br br: 1.0
film: 0.7752135432746733
movie: 0.7109704641350211
one: 0.45067922198209326
see: 0.2279767417927344
time: 0.22404034166923947
character: 0.20631367706082124
make: 0.19923844808068333
story: 0.18575692086034784
show: 0.1630647319131419
good: 0.16085211485026243
way: 0.1565040650406504
great: 0.15344242050015436
much: 0.15076669754039312
even: 0.14966038900895337
think: 0.14711330657610375
love: 0.14531233919934136
well: 0.14137593907584645
really: 0.13900895338067304
people: 0.13877740043223216...

-The top words align with positive sentiments associated with movie enjoyment, good content, and positive expressions.

-Words like "character," "story," and "show" highlight the importance of well-developed characters and engaging narratives in positive reviews.

-Some words, like "one," "see," and "time," are neutral and may not directly convey sentiment. Contextual analysis is crucial.

-Positive expressions like "great," "love," and "good" contribute to the overall positive sentiment in the word cloud.

...Top words and their frequencies(for negative reviews):
br br: 1.0
movie: 0.8132909121354926
film: 0.6266775111834079
one: 0.40265052747410474
make: 0.2087649211779059
even: 0.20787981723799728
character: 0.1964930746597134
time: 0.1868765399612468
made: 0.14910413128244385
good: 0.14800373178958448
bad: 0.1462096021816616
story: 0.14549195033849247
really: 0.14355429036193573
people: 0.14159270865727341
way: 0.1372867975982585
think: 0.13300480826734923
look: 0.13219146951175753
scene: 0.13128244384374327
know: 0.12984714015740498...

- The top words align with negative sentiments associated with dissatisfaction with movies, characters, and aspects of the content.
    
- Words like "one," "make," and "even" are neutral and may be used in both positive and negative contexts. 
    
- The presence of "good" and "bad" suggests an expression of opinions regarding the quality of content in negative reviews.
    
- Some words, such as "character," "see," and "time," are present in both positive and negative word clouds. 

###Top 20 n-grams for poitive and negative review

...Top 20 2-grams in bigrams for positive reviews:
one best: 1360 times
even though: 864 times
year old: 850 times
ever seen: 793 times
first time: 762 times
main character: 687 times
new york: 687 times
good movie: 656 times
special effect: 652 times
see movie: 646 times
must see: 635 times
year ago: 630 times
great movie: 621 times
movie like: 614 times
look like: 602 times
real life: 602 times
well done: 559 times
film like: 551 times
movie ever: 530 times
horror film: 529 times

Top 20 3-grams in threegrams for positive reviews:
movie ever seen: 201 times
new york city: 154 times
film ever made: 133 times
one best movie: 132 times
one best film: 127 times
based true story: 109 times
best movie ever: 108 times
film ever seen: 103 times
movie ever made: 101 times
first time saw: 87 times
seen long time: 80 times
first saw movie: 76 times
well worth watching: 73 times
international film festival: 71 times
would love see: 68 times
many year ago: 67 times
highly recommend movie: 65 times
one time favorite: 65 times
feel good movie: 60 times
one favorite movie: 59 times

Top 20 2-grams in bigrams for negative reviews:
look like: 1774 times
ever seen: 1378 times
special effect: 1206 times
waste time: 1153 times
movie ever: 1075 times
year old: 1033 times
bad movie: 1027 times
low budget: 995 times
worst movie: 919 times
main character: 883 times
horror movie: 878 times
movie like: 849 times
much better: 822 times
one worst: 758 times
even though: 722 times
good movie: 673 times
horror film: 657 times
bad guy: 642 times
make movie: 635 times
watch movie: 629 times

Top 20 3-grams in threegrams for negative reviews:
worst movie ever: 605 times
movie ever seen: 516 times
worst film ever: 281 times
one worst movie: 260 times
film ever seen: 253 times
movie ever made: 209 times
waste time money: 153 times
one worst film: 145 times
film ever made: 132 times
low budget movie: 106 times
bad acting bad: 103 times
complete waste time: 100 times
low budget film: 91 times
worst movie seen: 85 times
really bad movie: 80 times
make look like: 75 times
low budget horror: 74 times
could much better: 73 times
movie look like: 72 times
good thing movie: 72 times...


- Positive reviews tend to focus on specific elements such as effects, character development, and narrative quality.

- Negative reviews often mention disappointment with aspects like plot coherence, acting, and overall production value.

- Certain phrases like "worst movie ever" and "best movie ever" are highly polarizing and frequently appear in negative and positive reviews, respectively.

- The mention of "new york city" in both positive and negative reviews suggests the importance of setting or context in shaping viewer opinions.

## Description of feature engineering

- I removed irrelevant characters from the 'review' column using a function called remove_irrelevant_chars.
- I tokenized the cleaned reviews using a function named tokenize_text.
- I filtered out stop words from the tokenized reviews using filter_stop_words.
- I then lemmatized the filtered tokens using lemmatize_tokens.
- After that, I removed all words less than 2 characters long from the lemmatized tokens.
- Throughout the process, I displayed progress updates using the tqdm library to provide visual feedback on the progress of the data preprocessing.
- Finally, I returned the preprocessed DataFrame, which includes the original 'review' column along with additional columns representing each preprocessing step.

### Comparing stemming vs lemmatization

...Original Review:
I remember when this was in theaters, reviews said it was horrible. Well, I didn't think it was that bad. It was amusing and had a lot of tongue-in-cheek humor concerning families around holiday time.<br /><br />Ben Affleck is a rich guy who needs to find a family for Christmas to please his girlfriend. He goes to visit the house he grew 
up in and strikes a deal to rent the family there for Christmas. I really liked the lawyer scene where they sign a contract. That was funny.<br /><br />So, he makes silly requests of the family and even writes scripts for them to read. Of course, the family has a hot daughter for the love interest. And he learns that the holidays aren't so bad after all.<br /><br />Also, the whole doo-dah act was funny, especially when they replaced the first one with a black guy, and the girlfriends's parents didn't even say anything about it. And the parts where doo-dah is hitting on his "supposed daughter." FINAL VERDICT: I thought it's worth checking out if you catch it on cable.     

Stemmed Review:
['rememb', 'theater', 'review', 'said', 'horribl', 'well', 'think', 'bad', 'amus', 'lot', 'tongu', 'cheek', 'humor', 'concern', 'famili', 'around', 'holiday', 'time', 'br', 'br', 'ben', 'affleck', 'rich', 'guy', 'need', 'find', 'famili', 'christma', 'pleas', 'girlfriend', 'goe', 'visit', 'hous', 'grew', 'strike', 'deal', 'rent', 'famili', 'christma', 'realli', 'like', 'lawyer', 'scene', 'sign', 'contract', 'funni', 'br', 'br', 'make', 'silli', 'request', 'famili', 'even', 'write', 'script', 'read', 'cours', 'famili', 'hot', 'daughter', 'love', 'interest', 'learn', 'holiday', 'bad', 'br', 'br', 'also', 'whole', 'doo', 'dah', 'act', 'funni', 'especi', 'replac', 'first', 'one', 'black', 'guy', 'girlfriend', 'parent', 'even', 'say', 'anyth', 'part', 'doo', 'dah', 'hit', 'suppos', 'daughter', 'final', 'verdict', 'thought', 'worth', 'check', 'catch', 'cabl']

Lemmatized Review:
['remember', 'theater', 'review', 'said', 'horrible', 'well', 'think', 'bad', 'amusing', 'lot', 'tongue', 'cheek', 'humor', 'concerning', 'family', 'around', 'holiday', 'time', 'br', 'br', 'ben', 'affleck', 'rich', 'guy', 'need', 'find', 'family', 'christmas', 'please', 'girlfriend', 'go', 'visit', 'house', 'grew', 'strike', 'deal', 'rent', 'family', 'christmas', 'really', 'liked', 'lawyer', 'scene', 'sign', 'contract', 'funny', 'br', 'br', 'make', 'silly', 'request', 'family', 'even', 'writes', 'script', 'read', 'course', 'family', 'hot', 'daughter', 'love', 'interest', 'learns', 'holiday', 'bad', 'br', 'br', 'also', 'whole', 'doo', 'dah', 'act', 'funny', 'especially', 'replaced', 'first', 'one', 'black', 'guy', 'girlfriend', 'parent', 'even', 'say', 'anything', 'part', 'doo', 'dah', 'hitting', 'supposed', 'daughter', 'final', 'verdict', 'thought', 'worth', 'checking', 'catch', 'cable']...

- In this specific case, lemmatization provides a more linguistically accurate representation of the original text compared to stemming.

- Lemmatization preserves the semantic meaning of words and results in a more interpretable text representation, which is beneficial for tasks like sentiment analysis or topic modeling.

- Stemming, while computationally faster and simpler, may produce non-words or less interpretable results, potentially affecting downstream analysis tasks.

### Comparing vectorizaions

...Classification Report for CountVectorizer:
              precision    recall  f1-score   support

    negative       0.89      0.88      0.89      5000
    positive       0.89      0.89      0.89      5000
    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Classification Report for TfidfVectorizer:
              precision    recall  f1-score   support

    negative       0.91      0.88      0.89      5000
    positive       0.89      0.91      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
...

For CountVectorizer:

- The precision, recall, and F1-score for the 'negative' class are all 0.89, indicating balanced performance in correctly identifying negative sentiment.
- The macro average and weighted average scores for precision, recall, and F1-score are consistent, suggesting that the model performs consistently across both classes.

For TfidfVectorizer:

- The model performs slightly better for the 'positive' class compared to the 'negative' class, as seen from the precision, recall, and F1-score metrics for both classes.
- The weighted average F1-score is 0.90, indicating a good balance between precision and recall across both classes.