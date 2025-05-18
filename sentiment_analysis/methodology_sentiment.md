## **Sentiment Analysis with Active/Transfer Learning**
- first we tried to apply some pre-trained models (specificare quali) but their preformance was unsatisfactory because:
    1. very unbalanced results
    2. by simply checking a sample of the labeled outputs it was clear that annotation made no sense and grasped no semantic intuition
- moreover, simple labels such as 'negative', 'positive' and 'neutral' are not enough to express the complexity of our data
- for this reason, we introduce manual labels which have been created by reading some tweets and trying to understand which classes we can identify
- labels are:
    1.
    2.
    3.
    4.
- to classify the whole dataset we perform **transfer learning / active learning**: 
    - we start by sampling 200 tweets and manually label them
    - we fine tune an existing model on predicting the label on the manually annotated tweets
    - we perform inference on a subset of the remaining unlabeled data, compute the average confidence on each prediction and select the 100 least confident tweets: these will be the next batch to label
    - after labeling the new tweets, we repeat from the first step and track our confidence values
    - we stop once we get that we have good confidence and labels seem to qualitatively make sense

After manually labeling 700 samples, the model seems to perform well (0.26 average uncertainty), so we decide to stop. Confidence is good also on the whole dataset (recall that we were considering only 5000 samples). To further improve the performance, we select the tweets whose label is predicted with more than 90% of confidence and we add them to the labeled dataset (these new data will be called __pseudo_labeled__). Then, we fine tune over this dataset (partially manually annotated, partially annotated by the machine). Then we perform inference again on the whole dataset and, out of the final predictions, we only keep labels over a certain threshold of confidence. The rest will be labeled as 'generic_tone'.
