library(RWeka)
library(dplyr)
library(stringi)
library(tm)
library(NLP)
library(slam)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)



# Batch size
batch_size = 10000
# Directory of English files
en_US_files <- c('en_US.twitter.txt',
                 'en_US.blogs.txt',
                 'en_US.news.txt')

# List of data from different sources
data_source <- c()
# Read data from blogs, news and twitter
for(i in 1:length(en_US_files)){
  # Initialize current data
  data <- c()
  # Establish current file connection
  connection <- file(en_US_files[[i]], 'r')
  # Read data in batch
  while(TRUE){
    lines = readLines(connection, n = batch_size, skipNul = TRUE, warn = TRUE)
    if(length(lines) == 0){
      break
    }
    data <- c(data, lines)
  }
  # Save data from the current source
  data_source[[i]] <- data
  # Close the current file connection
  close(connection)
}


# Data statistics
Source <- c('Twitter', 'Blogs', 'News')
# File size (Megabytes)
Size <- c(as.numeric(object.size(x = data_source[[1]])/(1024^2)), 
          as.numeric(object.size(x = data_source[[2]])/(1024^2)),
          as.numeric(object.size(x = data_source[[3]])/(1024^2)))
# Number of sentences
nSentences <- c(length(data_source[[1]]), 
                length(data_source[[2]]), 
                length(data_source[[3]]))
# Number of words
nWords <- c(sum(sapply(data_source[[1]], function(data) {str_count(data, '\\S+')})),
            sum(sapply(data_source[[2]], function(data) {str_count(data, '\\S+')})),
            sum(sapply(data_source[[3]], function(data) {str_count(data, '\\S+')})))

statistics <- data.frame(Source, Size, nSentences, nWords)
# Total in all the sources
total <- data.frame(Source = 'Total', 
                    Size = sum(Size), 
                    nSentences = sum(nSentences), 
                    nWords = sum(nWords))
statistics <- rbind(statistics, total)
statistics


# Joining data from different sources into a data frame
data <- c(data_source[[1]], data_source[[2]], data_source[[3]])
# Sampling only 10% of data
set.seed(0710)
sample_size = as.integer(length(data)*0.10)
data_sample <- data.frame(sample(x = data, size = sample_size))
colnames(data_sample) <- c('Text')

# Splitting data for training and test
proportion <- 0.7
set.seed(0710)
perm <- sample(sample_size)
train_index <- perm[1:round(proportion*sample_size)]
test_index <- perm[round(proportion*sample_size + 1):sample_size]
train <- data_sample[train_index,, drop = FALSE]
test <- data_sample[test_index,, drop = FALSE]

# Write subsample of the original data to a separate .csv file
write.csv(x = train, file = './en_US_train_0710.csv', row.names = FALSE)
write.csv(x = test, file = './en_US_test_0710.csv', row.names = FALSE)

# Read the training data sample
data <- read.csv2('./en_US_train_0710.csv', stringsAsFactors = FALSE)
# Bad words for profanity filtering
bad_words <- as.vector(t(read.table('./badwords.txt')))
# Profanity filtering
str_bad_words <- paste(bad_words, collapse = '|')
idxs <- which(grepl(str_bad_words, data$Text, ignore.case = TRUE))
cleaned_data <- data[-idxs,, drop = FALSE]
# Write subsample of the original data to a separate .csv file
write.csv(x = cleaned_data, file = './en_US_train_0710_cleaned.csv', row.names = FALSE)

# Read the data sample
data <- read.csv2('./en_US_train_0710_cleaned.csv', stringsAsFactors = FALSE)
# Construct a corpus
vector_source <- VectorSource(as.vector(data$Text))
corpus <- VCorpus(vector_source)
# Data preparation
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, PlainTextDocument)



# N-gram tokenizers (generic)
ngram <- function(N){ 
  N
  function(corpus) { 
    NGramTokenizer(corpus, Weka_control(min = N, max = N))
  }
}



# List of N-gram tokenizers
N <- 4
ngram_funcs <- lapply(1:N, ngram)
# Corpus as a data.frame
df_corpus <- data.frame(Text=unlist(sapply(corpus, `[`,'content')), stringsAsFactors = FALSE)
# Top 10 N-tokens
top10tokens = c()
for (i in 1:N){
  i_gram <- ngram_funcs[[i]](df_corpus$Text)
  freq <- table(i_gram)
  top10tokens[[i]] <- sort(freq, decreasing = TRUE)[1:10]
}


top10tokens

