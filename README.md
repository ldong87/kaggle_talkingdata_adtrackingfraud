# kaggle_talkingdata_adtrackingfraud
Summary for the [Kaggle competition about advertisement fraud click identification by Talking Data](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). 

In this competition, the training data includes 180 million+ click records spanning 3 days. Each click record includes the following features:

- ip: ip address of click.

- app: app id for marketing.

- device: device type id of user mobile phone.

- os: os version id of user mobile phone.

- channel: channel id of mobile ad publisher.

- click_time: timestamp of click.

- attributed_time: if user download the app after click an ad, this is the time of the app being downloaded.

- is_attributed: the target to be predicted, indicating the app was downloaded.

This is a binary classification problem and the output is the probability of the app to be downloaded. We define that it is a "fraud click", when someone clicks an ad and doesn't download the corresponding app for marketing. Because the app owners pay the ad publishers / channels according to the number of clicks, the ad publisher can falsify large amount of "fraud clicks" to increase price. In order to better organize the advertisement money, in this challenge, we are asked to identify those clicks with low app download probability, i.e. fraud clicks.

Unlike many other Kaggle competitions which ultimately reduce to model emsembling competitions, this competition is more about feature engineering. We notice that, out of the 7 features, 5 of them are categorical and the other two are timestamps. We convert one timestamp feature to two categorical feature, day and hour. Since the aim is to identify fraud clickers, it is natural to try to pin down the fraudsters. The norm is to use the "ip, device, os" triplet to identify a unique clicker. We can groupby the triplet and count the unique app downloads. Similarly, we can create several count features by groupby different combinations of the categorical features. The count features include sum, mean and variance of counts. We created about 15 count features based on intuition.

We also created click time lag features, because fraud clicks may have very short lag. We created the lag features by groupby "ip, device, os and app". We used both forward and backward time lags.

We fed those features together with the original features into Light Gradient Boosting Machine (LGBM). We tune the hyperparameters usign a downsampled dataset since the whole data is too large, train data=180 million, test data=18 million. It takes about 10 hours to train and predict the model on the whole data with 48 threads without any GPUs. This model placed me to top 10% on public leader board. However, I forgot to select this submission as my final record unfortunately. Therefore, my performance on private leader board dropped to top 17%. My best performer on public leader board should win me a bronze medal with top 10%.

Now I'll outline some lessons I learned by reading the top teams' solutions.

- When a very powerful machine is available, we can create all combination of counts features by brutal force. Then we can choose to do feature selection or not. Good explanations: [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475).

- Create features with unsupervised learning. This acturally can be considered as a general way of model ensembling, except the model here is unsupervised learned. Because the features are all categorical. We can very easily relate this to topic modeling in NLP tasks. Therefore, we can use PCA, positive matrix factorization and latent dirichlet allocation to generate features from those counts data. Good explanations: [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475) and [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56283).

- Again inspired by the word2vec in NLP, we can try to encode cat2vec on these categorical variables.

- Since the click data has time pattern in it, we can use sequence model like RNN. It is easy to think of this, but it is very hard not easy to actually make this work. A very detailed explanation can be found [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262). Actually, for click series data, as long as we have a lot of them, we can always find ways to use RNN, see [an example from Airbnb](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e).

