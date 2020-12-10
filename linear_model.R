#
# title: MovieLens: a movie recommendation system
# author: Rob Meekings
# date: 10/12/2020
#

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                           title = as.character(title),
#                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Define function to calc rmse, use to express goodness of fit
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
# Record the time at which model fitting starts 
Sys.time()

##########################################################
# Model fitting
##########################################################

# Get mean rating for all movies in our training set
mu <- floor(mean(edx$rating)*1000) / 1000

# Get mean residuals by movie (good / bad movie)
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Get mean residuals by user (positive / negative reviewer)
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Get predictions based on these variables
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Check the error term
RMSE(predicted_ratings, validation$rating)

# Load libraries for time and date processing
library(lubridate)
library(tidyverse)

# Calculate when half scores were first used
first_half_dates <- edx %>% 
  group_by(rating) %>%
  summarize(dt = min(as_datetime(timestamp))) %>%
  ungroup() %>%
  filter(rating != floor(rating)) 

# Store the date of the first half score
first_half_date <- min(first_half_dates$dt)

# Add an indicator variable as to whether half scores were available
edx$halves <- ifelse(as_datetime(edx$timestamp) < first_half_date, 0, 1)

# Get mean residuals for the half score indicator
halves_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(halves) %>%
  summarize(b_h = mean(rating - mu - b_i - b_u))

# Add an indicator variable as to whether half scores were available
validation$halves <- ifelse(as_datetime(validation$timestamp) < first_half_date, 0, 1)

#Apply the model with the half score indciator parameter 
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  mutate(pred = mu + b_i + b_u + b_h) %>%
  pull(pred)

#Report the RMSE value
RMSE(predicted_ratings, validation$rating)

# Create a distinct list of movies 
movies <- edx %>% 
  group_by(movieId, title, genres) %>%
  summarize(num=n(), avg_rating=mean(rating)) %>%
  ungroup() %>%
  select(movieId, title, genres, num, avg_rating)

# Create a distinct list of genres
genre_list <- movies %>%
  group_by(genres) %>%
  summarize(movies=sum(num)) %>%
  ungroup() %>%
  select(genres) 

# Split the genres by pipe (|)
g <- str_split(genre_list$genres, "\\|")

# Count how many elements we've split into, get the max
n <- 0
for (i in 1:length(g)) {
  n <- ifelse(length(g[[i]]) > n, length(g[[i]]), n)
}

# Now create a matrix, row per genre, column per genre element
m <- matrix(data="", nrow=length(g), ncol=n)
# Loop over rows
for (i in 1:length(g)) {
  # Loop over cols
  for (j in 1:n) {
    # if the i'th row has j or more elements place the jth 
    # element into cell i,j of the matrix m
    m[i,j] <- ifelse(length(g[[i]]) >= j, str_replace(g[[i]][[j]], "-", "."), "")
    
  }
}
# Use set union to get a distinct list, use relative col refs
# to avoid naming cols, etc
cats <- union(union(union(m[,1], m[,2]),
                    union(m[,3], m[,4])),
              union(union(m[,5], m[,6]),
                    union(m[,7], m[,8]))) %>% 
  .[-length(.)] %>% # drop the last entry ""
  .[-1]             # drop the first entry "(no genres listed)"

#Create a variable that indicates if one of the columns matches the category
categories <- sapply(cats, function(cat){
  sign(ifelse(m[,1]==cat, 1, 0) + ifelse(m[,2] == cat, 1, 0) +
         ifelse(m[,3]==cat, 1, 0) + ifelse(m[,4] == cat, 1, 0) +
         ifelse(m[,5]==cat, 1, 0) + ifelse(m[,6] == cat, 1, 0) +
         ifelse(m[,7]==cat, 1, 0) + ifelse(m[,8] == cat, 1, 0))
})
# Assign names to the category list
names(categories) = cats
# Add the category indicators to the genre list
genre_list <- genre_list %>% mutate(data.frame(categories))
# Add the category indicators to the movie list
edx_genres <- inner_join(edx, genre_list, by="genres")
# Add the category indicators to the movie list
validation <- inner_join(validation, genre_list, by="genres")

# Get mean residuals for the Film Noir indicator
noir_avgs <- edx_genres %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  group_by(Film.Noir) %>%
  summarize(b_n = mean(rating - mu - b_i - b_u - b_h))

# Get mean residuals for the Drama indicator
drama_avgs <- edx_genres %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  group_by(Drama) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u - b_h - b_n))

# Get mean residuals for the Action indicator
action_avgs <- edx_genres %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  left_join(drama_avgs, by='Drama') %>%
  group_by(Action) %>%
  summarize(b_a = mean(rating - mu - b_i - b_u - b_h - b_n - b_d))

# Get mean residuals for the Crime indicator
crime_avgs <- edx_genres %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  left_join(drama_avgs, by='Drama') %>%
  left_join(action_avgs, by='Action') %>%
  group_by(Crime) %>%
  summarize(b_c = mean(rating - mu - b_i - b_u - b_h - b_n - b_d - b_a))

# Get mean residuals for the Comedy indicator
comedy_avgs <- edx_genres %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  left_join(drama_avgs, by='Drama') %>%
  left_join(action_avgs, by='Action') %>%
  left_join(crime_avgs, by='Crime') %>%
  group_by(Comedy) %>%
  summarize(b_k = mean(rating - mu - b_i - b_u - b_h - b_n - b_d - b_a - b_c))

# Apply the model with the category parameters
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  left_join(drama_avgs, by='Drama') %>%
  left_join(action_avgs, by='Action') %>%
  left_join(crime_avgs, by='Crime') %>%
  left_join(comedy_avgs, by='Comedy') %>%
  mutate(pred = mu + b_i + b_u + b_h + b_n + b_d + b_a + b_c + b_k) %>%
  pull(pred)

#Report the RMSE for this model
RMSE(predicted_ratings, validation$rating)

#Define a regular expression for picking release years
year_regex <- "\\(\\d{4}\\)$"

# Extract years and remove from titles
edx_genres <- edx_genres %>% 
  mutate(release_year = 
           str_replace_all(str_extract(title, year_regex), "\\(|\\)",""),
         title = str_replace(title, year_regex, ""))

# Get mean residuals for the release year variable
ryr_avgs <- edx_genres %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  left_join(drama_avgs, by='Drama') %>%
  left_join(action_avgs, by='Action') %>%
  left_join(crime_avgs, by='Crime') %>%
  group_by(release_year) %>%
  summarize(b_r = mean(rating - mu - b_i - b_u - b_h - b_n - b_d - b_a - b_c))

# Extract years and remove from titles
validation <- validation %>% 
  mutate(release_year = 
           str_replace_all(str_extract(title, year_regex), "\\(|\\)",""),
         title = str_replace(title, year_regex, ""))

# Apply the model with the release year parameters
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(halves_avgs, by='halves') %>%
  left_join(noir_avgs, by='Film.Noir') %>%
  left_join(drama_avgs, by='Drama') %>%
  left_join(action_avgs, by='Action') %>%
  left_join(crime_avgs, by='Crime') %>%
  left_join(ryr_avgs, by='release_year') %>%  
  mutate(pred = mu + b_i + b_u + b_h + b_n + b_d + b_a + b_c + b_r) %>%
  pull(pred)

#Report the RMSE for this model
RMSE(predicted_ratings, validation$rating)

# Record the time at which model fitting ends 
Sys.time()
